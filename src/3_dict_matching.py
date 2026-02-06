import pandas as pd
import pickle
import regex as re
from tqdm import tqdm
from src.utils.trie_dict import TrieNode
from rapidfuzz.distance import Levenshtein
from collections import defaultdict


def get_graphemes(text):
    return re.findall(r'\X', text)

def greedy_match_line(trie_root, text, min_len=4, max_len=30, exclude_spans=[]):
    graphemes = get_graphemes(text)
    i = 0
    matches = []

    exclude_mask = [False] * len(graphemes)
    # Convert exclude character spans to grapheme-level indices
    pos = [0]
    for g in graphemes:
        pos.append(pos[-1] + len(g))
    for s, e in exclude_spans:
        for i_g, (start, end) in enumerate(zip(pos[:-1], pos[1:])):
            if not (e <= start or s >= end):  # overlap
                exclude_mask[i_g] = True

    while i < len(graphemes):
        if exclude_mask[i]:
            i += 1
            continue

        node = trie_root
        longest = None

        for j in range(i, min(len(graphemes), i + max_len)):
            if exclude_mask[j]:
                break

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
                if span_len >= min_len:
                    longest = (i, j + 1)

        if longest:
            matches.append((pos[longest[0]], pos[longest[1]]))  # convert back to char spans
            for k in range(longest[0], longest[1]):
                exclude_mask[k] = True
            i = longest[1]
        else:
            i += 1

    return matches

def find_dict_matches_on_unmatched_parts(csv_path, dict_trie_path, output_csv, min_len=4, max_len=30):

    def get_graphemes(text):
        return re.findall(r'\X', text)

    def greedy_match_line(trie_root, text, min_len=4, max_len=30, exclude_spans=[]):
        graphemes = get_graphemes(text)
        i = 0
        matches = []

        # char position for each grapheme
        pos = [0]
        for g in graphemes:
            pos.append(pos[-1] + len(g))

        # convert exclude spans into grapheme mask
        exclude_mask = [False] * len(graphemes)
        for s, e in exclude_spans:
            for i_g, (start, end) in enumerate(zip(pos[:-1], pos[1:])):
                if not (e <= start or s >= end):  # overlap
                    exclude_mask[i_g] = True

        while i < len(graphemes):
            if exclude_mask[i]:
                i += 1
                continue

            node = trie_root
            longest = None

            for j in range(i, min(len(graphemes), i + max_len)):
                if exclude_mask[j]:
                    break

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
                matches.append((pos[longest[0]], pos[longest[1]]))  # back to char offsets
                for k in range(longest[0], longest[1]):
                    exclude_mask[k] = True
                i = longest[1]
            else:
                i += 1

        return matches

    def get_exclude_spans(text, matched_strs):
        spans = []
        for word in matched_strs.split("|"):
            if not word:
                continue
            start = 0
            while True:
                idx = text.find(word, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(word)))
                start = idx + len(word)
        return spans

    df = pd.read_csv(csv_path)
    with open(dict_trie_path, "rb") as f:
        dict_trie = pickle.load(f)

    dict_words_gt, dict_cov_gt = [], []
    dict_words_pred, dict_cov_pred = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt = str(row.get("ground_truth", "") or "")
        pred = str(row.get("prediction", "") or "")
        gt_lemma_matches = str(row.get("gt_words", "") or "")
        pred_lemma_matches = str(row.get("pred_words", "") or "")

        # --- Ground Truth ---
        exclude_gt = get_exclude_spans(gt, gt_lemma_matches)
        spans_gt = greedy_match_line(dict_trie, gt, min_len, max_len, exclude_spans=exclude_gt)
        words_gt = [gt[s:e] for s, e in spans_gt]
        cov_gt = sum(e - s for s, e in spans_gt) / len(gt) if len(gt) > 0 else 0.0

        # --- Prediction ---
        exclude_pred = get_exclude_spans(pred, pred_lemma_matches)
        spans_pred = greedy_match_line(dict_trie, pred, min_len, max_len, exclude_spans=exclude_pred)
        words_pred = [pred[s:e] for s, e in spans_pred]
        cov_pred = sum(e - s for s, e in spans_pred) / len(pred) if len(pred) > 0 else 0.0

        dict_words_gt.append("|".join(words_gt))
        dict_cov_gt.append(cov_gt)

        dict_words_pred.append("|".join(words_pred))
        dict_cov_pred.append(cov_pred)

    df["dict_matches_gt"] = dict_words_gt
    df["dict_matches_pred"] = dict_words_pred
    df["dict_coverage_gt"] = dict_cov_gt
    df["dict_coverage_pred"] = dict_cov_pred

    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

    print(f"GT coverage   (mean): {sum(dict_cov_gt)/len(dict_cov_gt):.3f}")
    print(f"PRED coverage (mean): {sum(dict_cov_pred)/len(dict_cov_pred):.3f}")

find_dict_matches_on_unmatched_parts(
    csv_path="results/lemma_matches.csv",
    dict_trie_path="trie/dictionary_trie.pkl",
    output_csv="results/lemma_matches_with_dict_matches.csv",
    min_len=3,
    max_len=30
)





# analysis
df = pd.read_csv("results/lemma_matches_with_dict_matches.csv") # importing the saved csv file from the same code above
with open("trie/dictionary_trie.pkl", "rb") as f:
    trie = pickle.load(f)

min_len = 3
max_len = 30
def spans_from_word_list(text, words_str):
    if not isinstance(words_str, str):
        return []
    spans = []
    for w in words_str.split("|"):
        if not w:
            continue
        start = 0
        while True:
            idx = text.find(w, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(w)))
            start = idx + len(w)
    return spans

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


matched_ops = matched_errors = unmatched_ops = unmatched_errors = 0

for _, row in df.iterrows():
    gt = str(row.get("ground_truth", ""))
    pred = str(row.get("prediction", ""))

    lemma_spans = spans_from_word_list(pred, row["pred_words"])
    lemma_mask = char_mask(len(pred), lemma_spans)
    
    dict_spans = spans_from_word_list(pred, row["dict_matches_pred"])
    dict_mask = char_mask(len(pred), dict_spans)

    dict_only_mask = [
        d and not l
        for d, l in zip(dict_mask, lemma_mask)
    ]

    ops = Levenshtein.editops(gt, pred)
    pred_error_idx = set()
    delete_errors = {}

    for tag, src, dest in ops:
        if tag in ("replace", "insert") and 0 <= dest < len(pred):
            pred_error_idx.add(dest)
        elif tag == "delete":
            delete_errors[dest] = delete_errors.get(dest, 0) + 1

    for j in range(len(pred)):
        if dict_only_mask[j]:
            matched_ops += 1
            if j in pred_error_idx:
                matched_errors += 1
        elif not lemma_mask[j]:
            unmatched_ops += 1
            if j in pred_error_idx:
                unmatched_errors += 1


    for dest, count in delete_errors.items():
        j = min(dest, len(pred) - 1) if pred else 0
        if dict_only_mask[j]:
            matched_errors += count
        elif not lemma_mask[j]:
            unmatched_errors += count

total_errors = matched_errors + unmatched_errors

print(f"Matched characters : {matched_ops}")
print(f"Unmatched characters : {unmatched_ops}")
print(f"Total errors      : {total_errors}")
print(f"Total errors in matched: {matched_errors}")
print(f"Total errors in unmatched: {unmatched_errors}")
# print(f"Error rate in matched : {matched_errors / matched_ops:.4f}")
# print(f"Error rate in unmatched : {unmatched_errors / unmatched_ops:.4f}")

print("Matched error rate  : ", matched_errors / total_errors)
print("Unmatched error rate: ", unmatched_errors / total_errors)

###########################################
# saving unmatched_errors to a file for later analysis
def add_spaces_around_spans(text, spans):
    """
    Wrap matched spans with spaces, without losing characters.
    """
    spans = sorted(spans)
    out = []
    last = 0
    for s, e in spans:
        out.append(text[last:s])
        out.append(" ")
        out.append(text[s:e])
        out.append(" ")
        last = e
    out.append(text[last:])
    return "".join(out)


def remove_spans_with_separator(text, spans, sep=" "):
    """
    Remove spans but insert a separator at each removal site
    so surrounding text chunks never merge.
    """
    if not spans:
        return text

    spans = sorted(spans)
    out = []
    last = 0

    for s, e in spans:
        # left context
        out.append(text[last:s])

        # separator (avoid stacking many)
        if out and not out[-1].endswith(sep):
            out.append(sep)

        last = e

    # tail
    out.append(text[last:])

    return "".join(out)


SEP = "--"   # or " - "

gt_unmatched = []
pred_unmatched = []

for _, row in df.iterrows():
    gt = str(row.get("ground_truth", ""))
    pred = str(row.get("prediction", ""))

    # --- GT spans ---
    gt_spans = (
        spans_from_word_list(gt, row.get("gt_words", "")) +
        spans_from_word_list(gt, row.get("dict_matches_gt", ""))
    )

    # --- PRED spans ---
    pred_spans = (
        spans_from_word_list(pred, row.get("pred_words", "")) +
        spans_from_word_list(pred, row.get("dict_matches_pred", ""))
    )

    gt_unmatched.append(
        remove_spans_with_separator(gt, gt_spans, sep=SEP)
    )
    pred_unmatched.append(
        remove_spans_with_separator(pred, pred_spans, sep=SEP)
    )


df["gt_unmatched_text"] = gt_unmatched
df["pred_unmatched_text"] = pred_unmatched

df.to_csv(
    "results/lemma_matches_with_dict_matches_unmatched.csv",
    index=False
)