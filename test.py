import pandas as pd
import pickle
from rapidfuzz.distance import Levenshtein
from src.utils.trie_dict import TrieNode
import regex as re
from rapidfuzz.distance import Levenshtein


min_length = 4
df = pd.read_csv("results/lemma_matches.csv")

# Get total character counts
df["gt_len"] = df["ground_truth"].astype(str).str.len()
df["pred_len"] = df["prediction"].astype(str).str.len()

total_gt_chars = df["gt_len"].sum()
total_pred_chars = df["pred_len"].sum()

print("=== TOTAL CHARACTER COUNTS ===")
print(f"Total GT characters   : {total_gt_chars}")
print(f"Total Pred characters : {total_pred_chars}")
print(f"Avg GT line length    : {df['gt_len'].mean():.2f}")
print(f"Avg Pred line length  : {df['pred_len'].mean():.2f}")



# Compute total edit distance and total GT length
def length_bin(n):
    if n == 3 or n==4:
        return "3-4"
    elif n==5 or n==6:
        return "5-6"
    elif n==7 or n==8:
        return "7-8"
    elif n >= 9:
        return "9>="
    

total_edit_distance = 0
total_gt_chars = 0

for _, row in df.iterrows():
    gt = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""
    pred = str(row["prediction"]) if pd.notna(row["prediction"]) else ""

    total_edit_distance += Levenshtein.distance(gt, pred)
    total_gt_chars += len(gt)

cer_overall = total_edit_distance / total_gt_chars if total_gt_chars > 0 else 0.0

print("\n=== OVERALL CER (no masking) ===")
print(f"Total edit distance: {total_edit_distance}")
print(f"Total GT characters: {total_gt_chars}")
print(f"CER                : {cer_overall:.4f}")

# handles devanagari graphemes
def get_graphemes(text):
    return re.findall(r'\X', text)

def greedy_match_line(trie_root, text, min_len=min_length, max_len=30):
    graphemes = get_graphemes(text)
    i = 0
    matches = []
    while i < len(graphemes):
        node = trie_root
        longest = None
        for j in range(i, min(len(graphemes), i + max_len)):
            g = graphemes[j]
            for ch in g:
                if ch not in node.children:
                    node = None
                    break
                node = node.children[ch]
            if node is None:
                break
            if node.entries and (j + 1 - i) >= min_len:
                longest = (i, j + 1)
        if longest:
            matches.append(longest)
            i = longest[1]
        else:
            i += 1
    # Grapheme to char span
    char_offsets = [0]
    for g in graphemes:
        char_offsets.append(char_offsets[-1] + len(g))
    return [(char_offsets[i], char_offsets[j]) for i, j in matches]

def char_mask(length, spans):
    mask = [False] * length
    for s, e in spans:
        for i in range(s, e):
            if i < length:
                mask[i] = True
    return mask

def compute_cer(gt, pred):
    if len(gt) == 0:
        return 0.0
    return Levenshtein.distance(gt, pred) / len(gt)

def analyze_inside_outside_cer(df, trie, min_len=min_length, max_len=30):
    inside_errors = 0
    inside_chars = 0
    outside_errors = 0
    outside_chars = 0

    for _, row in df.iterrows():
        gt = str(row["ground_truth"])
        pred = str(row["prediction"])
        pred_spans = greedy_match_line(trie, pred, min_len, max_len)
        mask = char_mask(len(pred), pred_spans)

        ops = Levenshtein.editops(gt, pred)

        for tag, src, dest in ops:
            if tag in ("insert", "replace") and 0 <= dest < len(pred):
                if mask[dest]:
                    inside_errors += 1
                else:
                    outside_errors += 1
            elif tag == "delete":
                dest_idx = min(dest, len(pred) - 1) if pred else 0
                if mask[dest_idx]:
                    inside_errors += 1
                else:
                    outside_errors += 1

        # Count total characters inside vs outside
        for i in range(len(pred)):
            if mask[i]:
                inside_chars += 1
            else:
                outside_chars += 1

    print("=== SIMPLE CER ANALYSIS ===")
    print(f"Inside matched spans:")
    print(f"  Errors: {inside_errors}")
    print(f"  Chars : {inside_chars}")
    print(f"  CER   : {inside_errors / inside_chars:.4f}" if inside_chars else "N/A")

    print(f"\nOutside matched spans:")
    print(f"  Errors: {outside_errors}")
    print(f"  Chars : {outside_chars}")
    print(f"  CER   : {outside_errors / outside_chars:.4f}" if outside_chars else "N/A")

    total_errors = inside_errors + outside_errors
    print(f"\nError distribution:")
    print(f"  Inside : {inside_errors} ({inside_errors / total_errors:.2%})")
    print(f"  Outside: {outside_errors} ({outside_errors / total_errors:.2%})")

# === Run ===

# df = pd.read_csv("results/lemma_matches.csv")
with open("trie/lemma_trie.pkl", "rb") as f:
    trie = pickle.load(f)

analyze_inside_outside_cer(df, trie, min_len=min_length, max_len=30)


cer_overall = total_edit_distance / total_gt_chars if total_gt_chars > 0 else 0.0

print("\n=== OVERALL CER (no masking) ===")
print(f"Total edit distance: {total_edit_distance}")
print(f"Total GT characters: {total_gt_chars}")
print(f"CER                : {cer_overall:.4f}")
print(f"Total character-level errors: {total_edit_distance}")
from collections import defaultdict

print("\n=== CER BY MATCHED SPAN LENGTH BIN ===")

bin_stats = defaultdict(lambda: {"errors": 0, "chars": 0})

for _, row in df.iterrows():
    gt = str(row["ground_truth"])
    pred = str(row["prediction"])
    pred_spans = greedy_match_line(trie, pred, min_len=min_length, max_len=30)
    mask = char_mask(len(pred), pred_spans)
    ops = Levenshtein.editops(gt, pred)

    # Map character positions to error
    error_locs = set()
    for tag, src, dest in ops:
        if tag in ("insert", "replace") and 0 <= dest < len(pred):
            error_locs.add(dest)
        elif tag == "delete":
            idx = min(dest, len(pred) - 1) if pred else 0
            error_locs.add(idx)

    for start, end in pred_spans:
        bin_name = length_bin(end - start)
        bin_stats[bin_name]["chars"] += (end - start)
        for i in range(start, end):
            if i in error_locs:
                bin_stats[bin_name]["errors"] += 1


for b in ("3-4", "5-6", "7-8", "9>="):
    e = bin_stats[b]["errors"]
    c = bin_stats[b]["chars"]
    cer = e / c if c > 0 else 0.0
    prec = 1 - cer
    print(f"{b:<6} {e:>8} {c:>8} {cer:>8.3f} {prec:>10.3f}")

# === Matched character coverage proportions ===
matched_gt_chars = 0
matched_pred_chars = 0

for _, row in df.iterrows():
    gt = str(row["ground_truth"])
    pred = str(row["prediction"])

    gt_spans = greedy_match_line(trie, gt, min_len=min_length, max_len=30)
    pred_spans = greedy_match_line(trie, pred, min_len=min_length, max_len=30)

    matched_gt_chars += sum(e - s for s, e in gt_spans)
    matched_pred_chars += sum(e - s for s, e in pred_spans)

total_gt_chars = df["gt_len"].sum()
total_pred_chars = df["pred_len"].sum()

prop_gt_matched = matched_gt_chars / total_gt_chars if total_gt_chars else 0.0
prop_pred_matched = matched_pred_chars / total_pred_chars if total_pred_chars else 0.0

print("\n=== MATCHED CHARACTER COVERAGE ===")
print(f"GT matched characters   : {matched_gt_chars}")
print(f"PRED matched characters : {matched_pred_chars}")
print(f"GT match proportion     : {prop_gt_matched:.3f}")
print(f"PRED match proportion   : {prop_pred_matched:.3f}")
