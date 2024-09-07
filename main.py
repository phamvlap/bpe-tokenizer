from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

sentences = [
    "aaabdaaabac",
    "Hi, I like to eat apples. How about you?",
]

for Tokenizer in [BasicTokenizer, RegexTokenizer, GPT4Tokenizer]:
    print(f"Using: {Tokenizer.__name__}")
    tokenizer_name = Tokenizer.__name__.lower().replace("tokenizer", "")

    if tokenizer_name == "gpt4":
        tokenizer = Tokenizer()
    else:
        tokenizer = Tokenizer()
        # tokenizer.train(text=text, vocab_size=256 + 3, verbose=True)
        tokenizer.load(f"models/{tokenizer_name}/{tokenizer_name}.model")

    for sent in sentences:
        ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(ids)

        print("Original text: ", sent)
        print("Encoded text: ", ids)
        print("Decoded text: ", decoded)
        print()
