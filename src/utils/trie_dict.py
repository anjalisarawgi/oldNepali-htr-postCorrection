import pandas as pd
import pickle
import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.entries = []  


def build_dictionary_trie(csv_path, word_col, min_len=2):
    df = pd.read_csv(csv_path)

    root = TrieNode()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        word = row.get(word_col)

        # can be disabled
        if len(word) < min_len:
            skipped += 1
            continue

        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        node.entries.append({"word_norm": word})

        inserted += 1

    print(f"Inserted {inserted} dictionary words")
    print(f"Skipped  {skipped} entries")

    return root



if __name__ == "__main__":
    DICT_CSV = "data/nepali-brihat-sabdakosh-processed-normalized.csv"  
    WORD_COL = "word_norm"

    trie = build_dictionary_trie( csv_path=DICT_CSV, word_col=WORD_COL, min_len=2)
    with open("trie/dictionary_trie.pkl", "wb") as f:
        pickle.dump(trie, f)

    print("Trie saved to: trie/dictionary_trie.pkl")
