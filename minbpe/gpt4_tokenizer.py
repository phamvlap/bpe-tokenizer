import os
import tiktoken

from minbpe import RegexTokenizer
from .utils import bytes_to_string
from .constants import GPT4_SPECIAL_TOKENS, MAX_BYTE_SIZE, SplitPattern


# Helper functions

"""
bpe: Byte Pair Encoding
Convert a token into a list of bytes [byte0, byte1]
    mergeable_ranks [dict[bytes, int]]: mergeable ranks of tokens
    token [bytes]: token (bytes) need to be merged
"""


def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    # Split token into a list of bytes
    parts = [bytes([t]) for t in token]

    while True:
        min_idx = None
        min_rank = None

        # Find the byte with the lowest rank
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_rank = rank
                min_idx = i
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        # Merge the two bytes (min_idx, min_idx + 1)
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )

    return parts


"""
Recover merges from mergeable ranks
"""


def recover_merges(mergeable_ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    merges = {}

    for token, rank in mergeable_ranks.items():
        # Skip single characters
        if len(token) == 1:
            continue
        # Find the pair of bytes that can be merges
        pair = tuple(bpe(mergeable_ranks=mergeable_ranks, token=token, max_rank=rank))
        idx0 = mergeable_ranks[pair[0]]
        idx1 = mergeable_ranks[pair[1]]

        merges[(idx0, idx1)] = rank

    return merges


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self) -> None:
        super().__init__(pattern=SplitPattern.GPT4_SPLIT_PATTERN)
        # Load the encoding from the tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)

        vocab = {idx: bytes([idx]) for idx in range(MAX_BYTE_SIZE)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        # byte_shuffle: byte (0-255) -> mergeable rank[byte]
        self.byte_shuffle = {
            i: mergeable_ranks[bytes([i])] for i in range(MAX_BYTE_SIZE)
        }
        # inverse_byte_shuffle: mergeable rank[byte] -> byte (0-255)
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be trained")

    def save(self, file_prefix: str) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be saved")

    def load(self, model_file: str) -> None:
        raise NotImplementedError("GPT4Tokenizer cannot be loaded")

    def encode_chunk(self, text_bytes: bytes) -> list[int]:
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        return super().encode_chunk(text_bytes)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes([self.inverse_byte_shuffle[b] for b in text_bytes])
        return text_bytes.decode(encoding="utf-8", errors="replace")

    def save_vocab(self, vocab_file: str) -> None:
        vocab = {
            idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(MAX_BYTE_SIZE)
        }
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        save_dir = vocab_file.rsplit("/", 1)[0]
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = bytes_to_string(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = bytes_to_string(vocab[idx0])
                    s1 = bytes_to_string(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
