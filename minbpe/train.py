import os
import time

from minbpe import Tokenizer, BasicTokenizer, RegexTokenizer, GPT4Tokenizer
from .constants import TokenizerType


def get_tokenizer(tokenizer_name: str) -> Tokenizer:
    tokenizer_name = tokenizer_name.lower()

    tokenizers = {
        TokenizerType.BASIC: BasicTokenizer,
        TokenizerType.REGEX: RegexTokenizer,
        TokenizerType.GPT4: GPT4Tokenizer,
    }

    return tokenizers.get(tokenizer_name, BasicTokenizer)


def train_bpe(input_path: str, output_dir: str, tokenizers: list[dict]):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} not found.")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tokenizer_dict in tokenizers:
        start = time.time()

        name = tokenizer_dict["name"]
        Tokenizer = get_tokenizer(tokenizer_dict["name"])
        tokenizer = Tokenizer()
        vocab_size = (
            tokenizer_dict["vocab_size"]
            if "vocab_size" in tokenizer_dict.keys()
            else 512
        )
        verbose = (
            tokenizer_dict["verbose"] if "verbose" in tokenizer_dict.keys() else True
        )

        tokenizer.train(text=text, vocab_size=vocab_size, verbose=verbose)

        prefix = os.path.join(output_dir, name, name)
        tokenizer.save(file_prefix=prefix)

        end = time.time()

        print("Completed training.")
        print("Training time: {:.2f} seconds.".format(end - start))
