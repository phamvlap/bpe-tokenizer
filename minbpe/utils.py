import unicodedata


# Statistic frequency of pairs of adjacent tokens in a list of ids
def get_statistics(ids: list[int], counts: dict = None) -> dict:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# Replace all occurrences of a pair of tokens with a single token (new_index)
def merge(ids: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i + 1 < len(ids) and ids[i + 1] == pair[1]:
            new_ids.append(new_index)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# Replace all control characters in a string with escape
# ord(c): convert a unicode character into its unicode code point (int)
# unicodedata.category(c): get the general category assigned to a unicode character
# ["\r", "\0", "\t"]: list of control characters (unicodedata.category() starts with "C")
def replace_control_chars(s: str) -> str:
    chars = []
    for c in s:
        if unicodedata.category(c)[0] != "C":
            chars.append(c)
        else:
            chars.append("\\u{:04x}".format(ord(c)))
    return "".join(chars)


# Render pretty token
def bytes_to_string(t: bytes) -> str:
    s = t.decode(encoding="utf-8", errors="replace")
    return replace_control_chars(s)
