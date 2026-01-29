import torch
from utils.trie_lemma import TrieNode
from utils.trie_dict import TrieNode
import regex as re
import pandas as pd
from tqdm import tqdm
import pickle

CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
VOWELS = set("अआइईउऊऋएऐओऔ")
MATRAS = set("ािीुूृेैोौ")
DEVANAGARI_SYMB = set("ँंः")
VIRMA = "्"


def get_graphemes(text):
    return re.findall(r'\X', text) 

def greedy_match_line(trie_root, text, min_len=4, max_len=30):
    graphemes = get_graphemes(text)
    i = 0
    matches = []

    while i < len(graphemes):
        node = trie_root
        longest_match = None

        for j in range(i, min(len(graphemes), i + max_len)):
            g = graphemes[j]
            for ch in g:
                if ch not in node.children:
                    node = None
                    break
                node = node.children[ch]
            if node is None:
                break

            if node.entries:
                span_len = j + 1 - i
                surface = "".join(graphemes[i:j+1])
                if span_len >= min_len:
                    longest_match = (i, j + 1)

        if longest_match:
            matches.append(longest_match)
            i = longest_match[1]
        else:
            i += 1

    # Convert grapheme spans back to character offsets
    char_matches = []
    pos = [0]
    for g in graphemes:
        pos.append(pos[-1] + len(g))
    for i, j in matches:
        char_matches.append((pos[i], pos[j]))

    return char_matches

def greedy_matches_with_lemmas(trie, text, min_len=4, max_len=30):
    matches = greedy_match_line(trie, text, min_len=min_len, max_len=max_len)

    surfaces = []
    lemmas = []

    for s, e in matches:
        surface = text[s:e]

        # re-walk trie to fetch lemma (safe + simple)
        node = trie
        for ch in surface:
            node = node.children[ch]

        lemma = node.entries[0]["lemma"] if node.entries else None

        surfaces.append(surface)
        lemmas.append(lemma)

    covered = sum(e - s for s, e in matches)
    total = len(text)
    coverage = covered / total if total > 0 else 0.0

    return matches, surfaces, lemmas, coverage

def analyze_row(trie, gt, pred, min_len=4, max_len=30):
    _, gt_words, gt_lemmas, gt_cov = greedy_matches_with_lemmas(
        trie, gt, min_len, max_len
    )
    _, pr_words, pr_lemmas, pr_cov = greedy_matches_with_lemmas(
        trie, pred, min_len, max_len
    )

    gt_set = set(gt_words)
    pr_set = set(pr_words)

    lost_words = sorted(list(gt_set - pr_set))
    gained_words = sorted(list(pr_set - gt_set))

    # lemma-level diff (parallel, additive)
    gt_lemma_set = set(gt_lemmas)
    pr_lemma_set = set(pr_lemmas)

    lost_lemmas = sorted(list(gt_lemma_set - pr_lemma_set))
    gained_lemmas = sorted(list(pr_lemma_set - gt_lemma_set))

    return {
        # EXISTING (unchanged)
        "gt_words": "|".join(gt_words),
        "pred_words": "|".join(pr_words),
        "gt_lemmas": "|".join(filter(None, gt_lemmas)),
        "pred_lemmas": "|".join(filter(None, pr_lemmas)),
        "gt_word_count": len(gt_words),
        "pred_word_count": len(pr_words),
        "gt_coverage": gt_cov,
        "pred_coverage": pr_cov,
        "coverage_delta": gt_cov - pr_cov,
        "lost_word_count": len(lost_words),
        "gained_word_count": len(gained_words),
        "lost_lemmas": "|".join(lost_lemmas), # lemmas in gt but not in pred
        "gained_lemmas": "|".join(gained_lemmas), # lemmas in pred but not in gt  
        "lost_words": "|".join(lost_words), # words in gt but not in pred
        "gained_words": "|".join(gained_words),    # words in pred but not in gt

    }

def run_analysis(input_csv, trie_path, output_csv,min_len, max_len):
    df = pd.read_csv(input_csv)

    with open(trie_path, "rb") as f:
        trie = pickle.load(f)

    out_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""
        pred = str(row["prediction"]) if pd.notna(row["prediction"]) else ""

        stats = analyze_row(trie, gt, pred, min_len=min_len, max_len=max_len)

        new_row = row.copy()
        for k, v in stats.items():
            new_row[k] = v

        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    INPUT_CSV = "data/predictions/predictions.csv" 
    run_analysis( input_csv=INPUT_CSV, trie_path="trie/lemma_trie.pkl", output_csv="results/lemma_matches.csv", min_len=4, max_len=30)