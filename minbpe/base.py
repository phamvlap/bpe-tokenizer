import os

from .utils import bytes_to_string
from .constants import MAX_BYTE_SIZE


class Tokenizer:
    """
    Default: vocab size of 256 (bytes), no merges, no patterns, no special tokens
        merges: dict[tuple[int, int], int]
        pattern: str
        special_tokens: dict[str, int]
        vocab: dict[int, bytes]
    """

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self.__build_vocab()

    def __build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(MAX_BYTE_SIZE)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode(encoding="utf-8")

        return vocab

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, s: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    """
    Save two files:
    - .model: contains version, pattern, special tokens, and merges tokens
    - .vocab: contains the vocabulary (pretty printed)
    """

    def save(self, file_prefix: str) -> None:
        save_dir = file_prefix.rsplit("/", 1)[0]
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_file = file_prefix + ".model"
        vocab_file = file_prefix + ".vocab"

        with open(model_file, "w") as f:
            f.write("minbpe v1.0\n")
            f.write(f"{self.pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")

            f.write(f"{len(self.merges)}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = bytes_to_string(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = bytes_to_string(self.vocab[idx0])
                    s1 = bytes_to_string(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    """
    Only load the .model file
    """

    def load(self, model_file: str) -> None:
        if not os.path.exists(model_file):
            raise ValueError(f"{model_file} not found.")

        idx = MAX_BYTE_SIZE

        with open(model_file, "r") as f:
            _version = f.readline().strip()
            self.pattern = f.readline().strip()

            num_special_tokens = int(f.readline().strip())
            for _ in range(num_special_tokens):
                token, cur_idx = f.readline().strip().split()
                self.special_tokens[token] = int(cur_idx)

            num_merges = int(f.readline().strip())
            for _ in range(num_merges):
                idx1, idx2 = map(int, f.readline().strip().split())
                self.merges[(idx1, idx2)] = int(idx)
                idx += 1

        self.vocab = self.__build_vocab()
