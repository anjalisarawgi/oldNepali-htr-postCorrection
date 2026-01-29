import pandas as pd
import pickle

class TrieNode:
    def __init__(self):
        self.children = {}
        self.entries = []

def build_lemma_trie(
    lemma_csv_path,
    surface_col="word_form_norm",
    lemma_col="lemma_norm",
):
    df = pd.read_csv(lemma_csv_path)
    df = df.drop_duplicates(subset=[surface_col, lemma_col])

    # Column fallbacks (important)
    if surface_col not in df.columns:
        if "word_form_clean" in df.columns:
            surface_col = "word_form_clean"
        else:
            surface_col = "word_form"

    if lemma_col not in df.columns:
        if "lemma_clean" in df.columns:
            lemma_col = "lemma_clean"
        else:
            lemma_col = "lemma"

    root = TrieNode()
    inserted = 0

    for _, row in df.iterrows():
        surface = str(row[surface_col]).strip()
        lemma = str(row[lemma_col]).strip()

        if not surface:
            continue

        node = root
        for ch in surface:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        # terminal node
        node.entries.append({
            "surface": surface,
            "lemma": lemma
        })
        inserted += 1

    print(f"Inserted {inserted} surface forms into trie")
    return root

def save_trie(trie_root, path="lemma_trie.pkl"):
    with open(path, "wb") as f:
        pickle.dump(trie_root, f)
    print("Trie saved to:", path)


def load_trie(path="lemma_trie.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_trie(trie_root, text):
    print(f"\nTesting trie on: {text}")
    node = trie_root

    for i, ch in enumerate(text):
        if ch not in node.children:
            print(f"Stopped at char '{ch}' (no further match)")
            break

        node = node.children[ch]

        if node.entries:
            for e in node.entries:
                print(
                    f"Match: '{e['surface']}'  "
                    f"(lemma='{e['lemma']}')  "
                    f"ends at index {i}"
                )

if __name__ == "__main__":
    LEMMA_CSV = "data/lemmas_normalized.csv"

    trie = build_lemma_trie(
        lemma_csv_path=LEMMA_CSV,
        surface_col="word_form_norm",
        lemma_col="lemma_norm",
    )

    save_trie(trie, "trie/lemma_trie.pkl")
