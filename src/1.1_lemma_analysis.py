import pandas as pd
from collections import defaultdict

df = pd.read_csv("results/lemma_matches.csv")

def split_words(x):
    if pd.isna(x) or x == "":
        return []
    return x.split("|")

df["gt_words_list"] = df["gt_words"].apply(split_words)
df["pred_words_list"] = df["pred_words"].apply(split_words)

df["has_lemma_gt"] = df["gt_word_count"] > 0
df["has_lemma_pred"] = df["pred_word_count"] > 0
df["has_lemma_any"] = df["has_lemma_gt"] | df["has_lemma_pred"]

df["coverage_abs_delta"] = df["coverage_delta"].abs()

print("Total lines:", len(df))
print("Lines with any lemma match:", df["has_lemma_pred"].sum())
print("Proportion of lines with matches:", round(df["has_lemma_pred"].mean(), 3))

print(f"GT coverage (mean):   {df['gt_coverage'].mean():.3f}")
print(f"PRED coverage (mean): {df['pred_coverage'].mean():.3f}")

# weighted coverage
df["gt_len"] = df["ground_truth"].astype(str).str.len()
df["pred_len"] = df["prediction"].astype(str).str.len()
total_gt_chars = df["gt_len"].sum()
total_pred_chars = df["pred_len"].sum()
weighted_gt_cov = (df["gt_coverage"] * df["gt_len"]).sum() / total_gt_chars
weighted_pred_cov = (df["pred_coverage"] * df["pred_len"]).sum() / total_pred_chars
print(f"GT weighted coverage   : {weighted_gt_cov:.3f}")
print(f"Pred weighted coverage : {weighted_pred_cov:.3f}")

#### check precision by lemma length
def length_bin(n):
    if n<=3:
        return "2-3"
    if n <= 5:
        return "4–5"
    elif n <= 8:
        return "6–8"
    else:
        return "9+"

def split_words(x):
    return x.split("|") if pd.notna(x) and x else []

stats = defaultdict(lambda: {"correct": 0, "total": 0})

for _, row in df.iterrows():
    gt_lemmas = set(split_words(row.get("gt_lemmas", "")))
    pred_lemmas = split_words(row.get("pred_lemmas", ""))

    for lemma in pred_lemmas:
        b = length_bin(len(lemma))
        stats[b]["total"] += 1
        if lemma in gt_lemmas:
            stats[b]["correct"] += 1
            
print("P(correct | lemma length)")
print(f"{'Length bin':<8} {'Correct':>8} {'Total':>8} {'Precision':>10}")
print("-" * 40)

for b in ("2-3", "4–5", "6–8", "9+"):
    c = stats[b]["correct"]
    t = stats[b]["total"]
    p = c / t if t > 0 else 0.0
    print(f"{b:<8} {c:>8} {t:>8} {p:>10.3f}")

# checking error rate / cer 
import pandas as pd
import pickle
from rapidfuzz.distance import Levenshtein
from utils.trie_dict import TrieNode
def analyze_error_localization(csv_path, trie_path, min_len=4, max_len=30):
    def greedy_match_line(trie_root, text):
        i = 0
        matches = []
        while i < len(text):
            node = trie_root
            longest = None
            for j in range(i, min(len(text), i + max_len)):
                ch = text[j]
                if ch not in node.children:
                    break
                node = node.children[ch]
                if node.entries and (j + 1 - i) >= min_len:
                    longest = (i, j + 1)
            if longest:
                matches.append(longest)
                i = longest[1]
            else:
                i += 1
        return matches

    def char_mask(length, spans):
        mask = [False] * length
        for s, e in spans:
            for i in range(s, e):
                if i < length:
                    mask[i] = True
        return mask

    df = pd.read_csv(csv_path)
    with open(trie_path, "rb") as f:
        trie = pickle.load(f)

    matched_ops = matched_errors = unmatched_ops = unmatched_errors = 0

    for _, row in df.iterrows():
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("prediction", ""))

        spans = greedy_match_line(trie, pred)
        mask = char_mask(len(pred), spans)

        ops = Levenshtein.editops(gt, pred)
        pred_error_idx = set()
        delete_errors = {}

        for tag, src, dest in ops:
            if tag in ("replace", "insert") and 0 <= dest < len(pred):
                pred_error_idx.add(dest)
            elif tag == "delete":
                delete_errors[dest] = delete_errors.get(dest, 0) + 1

        for j in range(len(pred)):
            if mask[j]:
                matched_ops += 1
                if j in pred_error_idx:
                    matched_errors += 1
            else:
                unmatched_ops += 1
                if j in pred_error_idx:
                    unmatched_errors += 1

        for dest, count in delete_errors.items():
            j = min(dest, len(pred) - 1) if pred else 0
            if mask[j]:
                matched_errors += count
            else:
                unmatched_errors += count

    total_errors = matched_errors + unmatched_errors

    print("\n=== ERROR LOCALIZATION ===")
    print(f"Matched ops (dict span) : {matched_ops}")
    print(f"Matched errors          : {matched_errors}")
    print(f"Error rate in dict span : {matched_errors / matched_ops:.4f}" if matched_ops else "N/A")
    print()
    print(f"Unmatched ops (outside) : {unmatched_ops}")
    print(f"Unmatched errors        : {unmatched_errors}")
    print(f"Error rate outside span : {unmatched_errors / unmatched_ops:.4f}" if unmatched_ops else "N/A")
    print()
    print("Error distribution:")
    print(f"  Inside dict spans  : {matched_errors} ({matched_errors / total_errors:.2%})")
    print(f"  Outside dict spans : {unmatched_errors} ({unmatched_errors / total_errors:.2%})")

analyze_error_localization(
    csv_path="results/lemma_matches.csv",
    trie_path="trie/lemma_trie.pkl",
    min_len=4,
    max_len=30
)