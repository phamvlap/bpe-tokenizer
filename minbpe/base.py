from minbpe.utils import clean_token


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
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}

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
                s = clean_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = clean_token(self.vocab[idx0])
                    s1 = clean_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    """
    Only load the .model file
    """

    def load(self, model_file: str) -> None:
        idx = 256

        with open(model_file, "r") as f:
            version = f.readline().strip()
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

        self.vocab = self._build_vocab()
