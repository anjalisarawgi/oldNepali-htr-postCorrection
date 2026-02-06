import pandas as pd
import pickle

class TrieNode:
    def __init__(self):
        self.children = {}
        self.stats = None   # terminal payload


def build_ngram_trie(
    ngram_csv_path,
    n_col="n",
    ngram_col="ngram",
    count_col="count",
    min_count=1,
):
    df = pd.read_csv(ngram_csv_path)

    root = TrieNode()
    inserted = 0

    for _, row in df.iterrows():
        ngram = str(row[ngram_col])
        n = int(row[n_col])
        count = int(row[count_col])

        if count < min_count or not ngram:
            continue

        node = root
        for ch in ngram:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        # terminal node
        node.stats = {
            "ngram": ngram,
            "n": n,
            "count": count,
        }
        inserted += 1

    print(f"Inserted {inserted} ngrams into trie")
    return root


def save_trie(trie_root, path):
    with open(path, "wb") as f:
        pickle.dump(trie_root, f)
    print("Trie saved to:", path)


def load_trie(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
ngram_trie = build_ngram_trie(
    ngram_csv_path="most_common_grapheme_ngrams.csv",
    min_count=20,   # VERY important: prune noise
)

save_trie(ngram_trie, "trie/ngram_trie.pkl")