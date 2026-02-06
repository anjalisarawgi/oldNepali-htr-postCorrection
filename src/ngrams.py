import collections
import regex as re
import pandas as pd

REMOVED_INVIS_CHARS = re.compile(r"[\u00AD\u200B\u200C\u200D]")

CORPUS_PATH = "data/label_corpus_oldNepali.txt"
NGRAM_SIZES = [3, 4, 5, 6 ]
COMMON_K = 1000
UNCOMMON_K = 500

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = REMOVED_INVIS_CHARS.sub("", text) # remive invisible characters
    text = re.sub(r"\s+", " ", text) # collapse multiple whitespace into single space

    return text.strip()


def graphemes(text):
    return re.findall(r"\X", text)


def grapheme_ngrams(text, n):
    g = graphemes(text)
    return ("".join(g[i:i+n]) for i in range(len(g) - n + 1))


def most_common_grapheme_ngrams(text, n, top_k=2000):
    counter = collections.Counter(grapheme_ngrams(text, n))
    return counter.most_common(top_k)


def least_common_grapheme_ngrams(text, n, bottom_k=2000):
    counter = collections.Counter(grapheme_ngrams(text, n))
    return sorted(counter.items(), key=lambda x: (x[1], x[0]))[:bottom_k]




text = load_corpus(CORPUS_PATH)

rows_common = []
rows_uncommon = []

for n in NGRAM_SIZES:
    counter = collections.Counter(grapheme_ngrams(text, n))

    for ng, cnt in counter.most_common(COMMON_K):
        rows_common.append({"ngram": ng, "n": n, "count": cnt})

    for ng, cnt in sorted(counter.items(), key=lambda x: (x[1], x[0]))[:UNCOMMON_K]:
        rows_uncommon.append({"ngram": ng, "n": n, "count": cnt})


pd.DataFrame(rows_common).to_csv("most_common_grapheme_ngrams.csv", index=False)
pd.DataFrame(rows_uncommon).to_csv("least_common_grapheme_ngrams.csv", index=False)

print("Saved both common and uncommon n-grams.")