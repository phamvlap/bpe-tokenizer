import os
import tiktoken

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer, Tokenizer


def unpack(text: str) -> str:
    if text.startswith("FILE:"):
        dirname = os.path.dirname(__file__)
        file = os.path.join(dirname, text[5:])

        content = ""
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        return content
    else:
        return text


# Special tokens
special_tokens = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}
# Test data
test_strings = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉",
    "FILE:data/text.txt",
    "aaabdaaabac",
]
specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()


# Test functions
def test_encode_decode(tokenizer_factory: Tokenizer, text: str) -> None:
    text = unpack(text)
    tokenizer = tokenizer_factory()

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert text == decoded, "Failed to encode and decode!"
    print("Passed!")


def test_special_token_regex(
    text: str,
    special_tokens: dict[str, int],
    allowed_special: bool | set = "none",
) -> None:
    tokenizer = RegexTokenizer()
    tokenizer.train(text=text, vocab_size=256 + 64, verbose=False)
    tokenizer.register_special_tokens(special_tokens=special_tokens)

    ids = tokenizer.encode(s=text, allowed_special=allowed_special)
    decoded = tokenizer.decode(ids)

    assert text == decoded, "Failed to encode and decode!"
    print("Passed!")

    prefix = "regex_tokenizer_tmp"
    tokenizer.save(file_prefix=prefix)

    tokenizer = RegexTokenizer()
    tokenizer.load(model_file=f"{prefix}.model")

    assert tokenizer.decode(ids) == text, "Failed to decode after loading tokenizer"
    assert (
        tokenizer.encode(text, allowed_special=allowed_special) == ids
    ), "Failed to encode after loading tokenizer"
    print("Passed!")

    for file in [f"{prefix}.model", f"{prefix}.vocab"]:
        if os.path.exists(file):
            os.remove(file)


def test_gpt4_tiktoken_equality(
    text: str,
    allowed_special: set | str = "none_raise",
) -> None:
    text = unpack(text)

    enc = tiktoken.get_encoding("cl100k_base")
    if allowed_special == "all":
        titoken_ids = enc.encode(text, allowed_special=allowed_special)
    else:
        titoken_ids = enc.encode(text)

    tokenizer = GPT4Tokenizer()
    gpt4_tokenizer_ids = tokenizer.encode(text, allowed_special=allowed_special)

    assert titoken_ids == gpt4_tokenizer_ids, "Failed to encode with GPT4Tokenizer!"
    print("Passed!")


if __name__ == "__main__":
    print("Testing encode and decode with plain text...")
    for tokenizer in [BasicTokenizer, RegexTokenizer, GPT4Tokenizer]:
        print(tokenizer.__name__)
        for text in test_strings:
            test_encode_decode(tokenizer, text)

    print("\nTesting RegexTokenizer with special tokens...")
    test_special_token_regex(
        text=llama_text,
        special_tokens=special_tokens,
        allowed_special="all",
    )

    print('\nTesting GPT4Tokenizer with "cl100k_base" for plain text...')
    for text in test_strings:
        test_gpt4_tiktoken_equality(text)

    print("\nTesting GPT4Tokenizer with special tokens in text...")
    test_gpt4_tiktoken_equality(text=specials_string, allowed_special="all")
