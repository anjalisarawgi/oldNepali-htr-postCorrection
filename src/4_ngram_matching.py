# import pandas as pd
# import pickle
# import regex as re
# from tqdm import tqdm


# ########################################
# # Grapheme utility
# ########################################

# def get_graphemes(text):
#     return re.findall(r"\X", text)


# ########################################
# # Trie node (same as saved)
# ########################################

# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.stats = None


# ########################################
# # Load existing ngram trie
# ########################################

# def load_ngram_trie(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# ########################################
# # Find ngram matches (sliding scan)
# ########################################
# def find_ngram_matches_nonoverlap(
#     trie_root,
#     text,
#     max_n=5
# ):
#     grams = get_graphemes(text)
#     matches = []

#     # grapheme index → char index
#     pos = [0]
#     for g in grams:
#         pos.append(pos[-1] + len(g))

#     i = 0
#     while i < len(grams):
#         best = None
#         best_j = None

#         # 🔥 longest-first
#         for n in range(min(max_n, len(grams) - i), 0, -1):
#             node = trie_root
#             ok = True

#             for j in range(i, i + n):
#                 g = grams[j]
#                 for ch in g:
#                     if ch not in node.children:
#                         ok = False
#                         break
#                     node = node.children[ch]
#                 if not ok:
#                     break

#             if ok and node and node.stats:
#                 best = {
#                     "ngram": node.stats["ngram"],
#                     "start": pos[i],
#                     "end": pos[i + n],
#                 }
#                 best_j = i + n - 1
#                 break  # ← stop at FIRST (longest) hit

#         if best is not None:
#             matches.append(best)
#             i = best_j + 1  # consume
#         else:
#             i += 1

#     return matches

# ########################################
# # Coverage computation
# ########################################

# def coverage_from_spans(text_len, spans):
#     if text_len == 0:
#         return 0.0

#     mask = [False] * text_len
#     for s, e in spans:
#         for i in range(s, min(e, text_len)):
#             mask[i] = True

#     return sum(mask) / text_len


# ########################################
# # Main pipeline
# ########################################

# def add_ngram_matches_and_coverage(
#     input_csv,
#     trie_path,
#     output_csv,
#     pred_col="pred_unmatched_text",
#     gt_col="gt_unmatched_text",
#     max_n=3
# ):
#     print("Loading input CSV...")
#     df = pd.read_csv(input_csv)

#     print("Loading ngram trie...")
#     trie = load_ngram_trie(trie_path)

#     pred_matches, pred_counts, pred_cov = [], [], []
#     gt_matches, gt_counts, gt_cov = [], [], []

#     print("Scanning unmatched text (PRED + GT)...")
#     for _, row in tqdm(df.iterrows(), total=len(df)):
#         pred_text = str(row.get(pred_col, ""))
#         gt_text = str(row.get(gt_col, ""))

#         pred_hits = find_ngram_matches_nonoverlap(trie, pred_text, max_n)
#         gt_hits = find_ngram_matches_nonoverlap(trie, gt_text, max_n)

#         pred_matches.append("|".join(h["ngram"] for h in pred_hits))
#         pred_counts.append(len(pred_hits))
#         pred_cov.append(
#             coverage_from_spans(
#                 len(pred_text),
#                 [(h["start"], h["end"]) for h in pred_hits]
#             )
#         )

#         gt_matches.append("|".join(h["ngram"] for h in gt_hits))
#         gt_counts.append(len(gt_hits))
#         gt_cov.append(
#             coverage_from_spans(
#                 len(gt_text),
#                 [(h["start"], h["end"]) for h in gt_hits]
#             )
#         )

#     # add columns (keep everything else intact)
#     df["pred_unmatched_ngram_matches"] = pred_matches
#     df["pred_unmatched_ngram_count"] = pred_counts
#     df["pred_unmatched_ngram_coverage"] = pred_cov

#     df["gt_unmatched_ngram_matches"] = gt_matches
#     df["gt_unmatched_ngram_count"] = gt_counts
#     df["gt_unmatched_ngram_coverage"] = gt_cov

#     print("Saving new CSV...")
#     df.to_csv(output_csv, index=False)
#     print("Saved:", output_csv)


# ########################################
# # Run
# ########################################


# output_csv = "results/lemma_matches_with_dict_matches_unmatched_ngram_matches_with_logits.csv"

# add_ngram_matches_and_coverage(
#     input_csv="results/lemma_matches_with_dict_matches_unmatched_with_logits_with_beta_cal.csv",
#     trie_path="trie/ngram_trie.pkl",
#     output_csv=output_csv,
#     max_n=3
# )

# print("\nComputing weighted mean coverage...")
# df = pd.read_csv(output_csv)

# pred_total_chars = 0
# pred_weighted_sum = 0.0

# gt_total_chars = 0
# gt_weighted_sum = 0.0

# for _, row in df.iterrows():
#     pred_text = str(row.get("pred_unmatched_text", ""))
#     gt_text = str(row.get("gt_unmatched_text", ""))

#     pred_len = len(pred_text)
#     gt_len = len(gt_text)

#     pred_cov = row.get("pred_unmatched_ngram_coverage", 0.0)
#     gt_cov = row.get("gt_unmatched_ngram_coverage", 0.0)

#     pred_total_chars += pred_len
#     pred_weighted_sum += pred_cov * pred_len

#     gt_total_chars += gt_len
#     gt_weighted_sum += gt_cov * gt_len

# weighted_pred_cov = (
#     pred_weighted_sum / pred_total_chars
#     if pred_total_chars > 0 else 0.0
# )

# weighted_gt_cov = (
#     gt_weighted_sum / gt_total_chars
#     if gt_total_chars > 0 else 0.0
# )

# print("\n===== Weighted Mean N-gram Coverage =====")
# print(f"Pred unmatched weighted coverage : {weighted_pred_cov:.4f}")
# print(f"GT unmatched weighted coverage   : {weighted_gt_cov:.4f}")
# print(
#     "Coverage ratio (Pred / GT)       : "
#     f"{(weighted_pred_cov / weighted_gt_cov) if weighted_gt_cov > 0 else 0.0:.4f}"
# )

# # ----------------------------------
# # Error analysis: ngram-matched vs unmatched
# # ----------------------------------
# # ----------------------------------
# # Error analysis: ngram-matched vs unmatched
# # ----------------------------------
# ########################################
# # Character mask utility
# ########################################

# def char_mask(length, spans):
#     """
#     Create a boolean mask of length `length`
#     where True indicates the character is covered by any span.
#     """
#     mask = [False] * length
#     for s, e in spans:
#         for i in range(s, min(e, length)):
#             mask[i] = True
#     return mask

# from rapidfuzz.distance import Levenshtein

# print("\nComputing error distribution over n-gram matched regions...")

# # 🔧 FIX: load trie in this scope
# trie = load_ngram_trie("trie/ngram_trie.pkl")

# matched_chars = unmatched_chars = 0
# matched_errors = unmatched_errors = 0

# for _, row in df.iterrows():
#     gt = str(row.get("ground_truth", ""))
#     pred = str(row.get("prediction", ""))

#     # ngram spans on FULL prediction text
#     ngram_spans = find_ngram_matches_nonoverlap(
#         trie,
#         pred,
#         max_n=3
#     )

#     ngram_mask = char_mask(
#         len(pred),
#         [(h["start"], h["end"]) for h in ngram_spans]
#     )
#     # edit operations
#     ops = Levenshtein.editops(gt, pred)
#     pred_error_idx = set()
#     delete_errors = {}

#     for tag, src, dest in ops:
#         if tag in ("replace", "insert") and 0 <= dest < len(pred):
#             pred_error_idx.add(dest)
#         elif tag == "delete":
#             delete_errors[dest] = delete_errors.get(dest, 0) + 1

#     # replace / insert attribution
#     for j in range(len(pred)):
#         if ngram_mask[j]:
#             matched_chars += 1
#             if j in pred_error_idx:
#                 matched_errors += 1
#         else:
#             unmatched_chars += 1
#             if j in pred_error_idx:
#                 unmatched_errors += 1

#     # delete attribution
#     for dest, count in delete_errors.items():
#         j = min(dest, len(pred) - 1) if pred else 0
#         if ngram_mask[j]:
#             matched_errors += count
#         else:
#             unmatched_errors += count


# total_errors = matched_errors + unmatched_errors

# print("\n===== N-GRAM ERROR DISTRIBUTION =====")
# print(f"Ngram-matched characters   : {matched_chars}")
# print(f"Ngram-unmatched characters : {unmatched_chars}")
# print(f"Total errors               : {total_errors}")
# print(f"Errors in ngram-matched     : {matched_errors}")
# print(f"Errors in ngram-unmatched   : {unmatched_errors}")

# print("\nError proportion:")
# print(
#     "Matched error ratio   :",
#     matched_errors / total_errors if total_errors else 0.0
# )
# print(
#     "Unmatched error ratio :",
#     unmatched_errors / total_errors if total_errors else 0.0
# )

# ########################
# import ast
# import numpy as np

# bins = [(i/10, (i+1)/10) for i in range(10)]

# stats = {
#     (lo, hi): {"eligible": 0, "correct": 0}
#     for lo, hi in bins
# }

# for _, row in df.iterrows():
#     pred_text = str(row.get("prediction", ""))
#     gt_ids = ast.literal_eval(str(row.get("gt_token_ids", "[]")))
#     pred_ids = ast.literal_eval(str(row.get("pred_token_ids", "[]")))
#     beta_probs = ast.literal_eval(str(row.get("cal_prob_beta", "[]")))

#     if not (len(gt_ids) == len(pred_ids) == len(beta_probs)):
#         continue

#     # ngram spans on FULL prediction text
#     spans = find_ngram_matches_nonoverlap(trie, pred_text, max_n=3)
#     mask = char_mask(
#         len(pred_text),
#         [(h["start"], h["end"]) for h in spans]
#     )

#     # token index → char index (left aligned)
#     char_idx = 0
#     token_char_pos = []
#     for g in get_graphemes(pred_text):
#         token_char_pos.append(char_idx)
#         char_idx += len(g)

#     for i, (gt, pr, p) in enumerate(zip(gt_ids, pred_ids, beta_probs)):
#         if i >= len(token_char_pos):
#             break

#         j = token_char_pos[i]
#         if not (0 <= j < len(mask)):
#             continue

#         if not mask[j]:
#             continue  # only ngram-matched tokens

#         for lo, hi in bins:
#             if lo <= p < hi:
#                 stats[(lo, hi)]["eligible"] += 1
#                 if gt == pr:
#                     stats[(lo, hi)]["correct"] += 1
#                 break

# import ast
# import numpy as np

# ########################################
# # Helpers
# ########################################

# def merge_masks(*masks):
#     out = masks[0][:]
#     for m in masks[1:]:
#         for i in range(len(out)):
#             out[i] = out[i] or m[i]
#     return out


# def token_char_positions(text):
#     pos = []
#     i = 0
#     for g in get_graphemes(text):
#         pos.append(i)
#         i += len(g)
#     return pos


# ########################################
# # Beta-bin accuracy computation
# ########################################

# bins = [(i / 10, (i + 1) / 10) for i in range(10)]

# bin_stats = {
#     (lo, hi): {"eligible": 0, "correct": 0}
#     for lo, hi in bins
# }

# for _, row in df.iterrows():
#     pred_text = str(row.get("prediction", ""))
#     if not pred_text:
#         continue

#     # token-level fields
#     gt_ids = ast.literal_eval(str(row.get("gt_token_ids", "[]")))
#     pred_ids = ast.literal_eval(str(row.get("pred_token_ids", "[]")))
#     beta_probs = ast.literal_eval(str(row.get("cal_prob_beta", "[]")))

#     if not (len(gt_ids) == len(pred_ids) == len(beta_probs)):
#         continue

#     ####################################
#     # N-gram mask
#     ####################################
#     ngram_spans = find_ngram_matches_nonoverlap(trie, pred_text, max_n=3)
#     ngram_mask = char_mask(
#         len(pred_text),
#         [(h["start"], h["end"]) for h in ngram_spans]
#     )

#     # dict mask 
#     dict_mask = [False] * len(pred_text)
#     try:
#         pred_spans = ast.literal_eval(str(row.get("pred_spans", "[]")))
#         if isinstance(pred_spans, list):
#             dict_mask = char_mask(len(pred_text), pred_spans)
#     except Exception:
#         pass


#     lemma_mask = [False] * len(pred_text)

#     pred_lemmas = str(row.get("pred_lemmas", "")).split("|")
#     gt_lemmas   = str(row.get("gt_lemmas", "")).split("|")

#     char_pos = 0
#     for p_lem, g_lem, g in zip(pred_lemmas, gt_lemmas, get_graphemes(pred_text)):
#         if p_lem == g_lem and p_lem != "":
#             for i in range(char_pos, char_pos + len(g)):
#                 if i < len(lemma_mask):
#                     lemma_mask[i] = True
#         char_pos += len(g)

#     structural_mask = merge_masks(ngram_mask, dict_mask, lemma_mask)


#     tok_char_pos = token_char_positions(pred_text)

#     for i, (gt, pr, p) in enumerate(zip(gt_ids, pred_ids, beta_probs)):
#         if i >= len(tok_char_pos):
#             break

#         j = tok_char_pos[i]
#         if j >= len(structural_mask):
#             continue

#         if not structural_mask[j]:
#             continue

#         for lo, hi in bins:
#             if lo <= p < hi:
#                 bin_stats[(lo, hi)]["eligible"] += 1
#                 if gt == pr:
#                     bin_stats[(lo, hi)]["correct"] += 1
#                 break


# print("\n===== STRUCTURE-CONDITIONED BETA BIN ACCURACY =====")
# print("(Token selected if ngram OR dict OR lemma)\n")

# for lo, hi in bins:
#     e = bin_stats[(lo, hi)]["eligible"]
#     c = bin_stats[(lo, hi)]["correct"]
#     acc = c / e if e else 0.0
#     print(f"Beta [{lo:.1f}, {hi:.1f}) | tokens={e:6d} | acc={acc:.4f}")