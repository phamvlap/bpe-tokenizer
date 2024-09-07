from minbpe.train import train_bpe
from minbpe.constants import TokenizerType


tokenizers = [
    {
        "name": TokenizerType.BASIC,
        "vocab_size": 1024,
        "verbose": True,
    },
    {
        "name": TokenizerType.REGEX,
        "vocab_size": 1024,
        "verbose": True,
    },
]

train_bpe(input_path="data/sample.txt", output_dir="models", tokenizers=tokenizers)
