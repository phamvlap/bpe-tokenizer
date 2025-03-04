from .base import Tokenizer
from .utils import get_statistics, merge
from .constants import MAX_BYTE_SIZE


class BasicTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        num_merges = vocab_size - MAX_BYTE_SIZE
        ids = list(text.encode(encoding="utf-8"))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(MAX_BYTE_SIZE)}

        print("Training Basic Tokenizer...")
        for i in range(num_merges):
            stats = get_statistics(ids)
            top_pair = max(stats, key=stats.get)
            idx = MAX_BYTE_SIZE + i
            ids = merge(ids=ids, pair=top_pair, new_index=idx)
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

    def encode(self, s: str) -> list[int]:
        tokens = list(s.encode(encoding="utf-8"))

        while len(tokens) > 1:
            stats = get_statistics(tokens)
            pair = min(
                stats,
                key=lambda pair: self.merges.get(pair, float("inf")),
            )
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(ids=tokens, pair=pair, new_index=idx)

        return tokens

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[idx] for idx in ids]
        return b"".join(tokens).decode(encoding="utf-8", errors="replace")
