import os
import time

from minbpe import BasicTokenizer, RegexTokenizer

vocab_size = 512

text = open("data/text.txt", "r", encoding="utf-8").read()
os.makedirs("models", exist_ok=True)

start = time.time()

for Tokenizer, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    tokenizer = Tokenizer()

    tokenizer.train(text=text, vocab_size=vocab_size, verbose=True)

    prefix = os.path.join("models", name)
    tokenizer.save(file_prefix=prefix)

end = time.time()

print("Completed training.")
print("Training time: {:.2f} seconds.".format(end - start))
