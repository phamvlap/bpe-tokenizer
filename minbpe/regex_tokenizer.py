import regex as re

from .base import Tokenizer
from .utils import get_statistics, merge
from .constants import SplitPattern, MAX_BYTE_SIZE

"""
pattern [str]: regex pattern to split text into tokens
compiled_pattern [re.Pattern]: compiled regex pattern
special_tokens [dict[str, int]]: special tokens (e.g. {'<|endoftext|>': 100257})
inverse_special_tokens [dict[int, str]]: inverse of special tokens
"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = None) -> None:
        super().__init__()
        self.pattern = SplitPattern.GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        num_merges = vocab_size - MAX_BYTE_SIZE

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode(encoding="utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(MAX_BYTE_SIZE)}

        print("Training Regex Tokenizer...")
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = get_statistics(chunk_ids, stats)

            top_pair = max(stats, key=stats.get)
            idx = MAX_BYTE_SIZE + i
            ids = [
                merge(ids=chunk_ids, pair=top_pair, new_index=idx) for chunk_ids in ids
            ]

            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            if verbose:
                print(
                    "Merge {:>4} / {:<4}: {:>10} -> {:>4} ({}) had {} occurrences".format(
                        i + 1,
                        num_merges,
                        str(top_pair),
                        idx,
                        str(vocab[idx]),
                        stats[top_pair],
                    )
                )
        self.merges = merges
        self.vocab = vocab

    def encode(self, s: str, allowed_special: str | set = "none_raise") -> list[int]:
        special_tokens = None
        if allowed_special == "all":
            special_tokens = self.special_tokens
        elif allowed_special == "none":
            special_tokens = {}
        elif allowed_special == "none_raise":
            special_tokens = {}
            assert all(
                token not in s for token in self.special_tokens
            ), "Error: Text contains special tokens"
        elif isinstance(allowed_special, set):
            special_tokens = {
                key: value
                for key, value in self.special_tokens.items()
                if key in allowed_special
            }
        else:
            raise ValueError(
                "allowed_special = {} not understood".format(allowed_special.items())
            )
        # If no special tokens, just use the ordinary encoding
        if not special_tokens:
            return self.encode_ordinary(s)
        # Create a regex pattern to match special tokens
        special_pattern = (
            "(" + "|".join(re.escape(k) for k in special_tokens.keys()) + ")"
        )
        # Split the text into chunks of ordinary text and special tokens
        special_chunks = re.split(pattern=special_pattern, string=s)

        ids = []
        for chunk in special_chunks:
            if chunk in special_tokens:
                ids.append(special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(text=chunk))

        return ids

    def decode(self, ids: list[int]) -> str:
        list_bytes = []

        for idx in ids:
            if idx in self.vocab:
                list_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                list_bytes.append(
                    self.inverse_special_tokens[idx].encode(encoding="utf-8")
                )
            else:
                raise ValueError("Unknown token: {}".format(idx))

        text_bytes = b"".join(list_bytes)
        return text_bytes.decode(encoding="utf-8", errors="replace")

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {
            idx: token for token, idx in special_tokens.items()
        }

    def encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)

        while len(ids) > 1:
            stats = get_statistics(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids=ids, pair=pair, new_index=idx)

        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode(encoding="utf-8")
            chunk_ids = self.encode_chunk(text_bytes=chunk_bytes)
            ids.extend(chunk_ids)

        return ids
