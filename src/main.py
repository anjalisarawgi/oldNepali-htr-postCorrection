from operator import is_
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import math
from collections import Counter

CSV_PATH = "data/predictions/predictions_with_calibrated_probs.csv"

df = pd.read_csv(CSV_PATH, converters={
    "pred_tokens": ast.literal_eval,
    "gt_tokens": ast.literal_eval, 
    "gt_token_ids": ast.literal_eval,
    "pred_token_ids": ast.literal_eval,
    "top3_ids": ast.literal_eval,
    "top3_probs": ast.literal_eval,
    "entropies": ast.literal_eval, 
    "pred_token_probs_cal": ast.literal_eval,
})


def pick_errors(prediction_tokens, groundtruth_tokens, prediction_tokens_ids, groundtruth_tokens_ids, top3_ids, top3_probs, entropies, pred_token_probs_cal, row):
    aligned = []
    m = SequenceMatcher(None, groundtruth_tokens, prediction_tokens)

    def add(i, j , tag): # i = gt, j = pred
        gt_tokens = groundtruth_tokens[i]  if i < len(groundtruth_tokens) else None
        pred_tokens = prediction_tokens[j] if j < len(prediction_tokens) else None
        gt_token_id = groundtruth_tokens_ids[i]  if i < len(groundtruth_tokens_ids) else None
        pred_token_id = prediction_tokens_ids[j] if j < len(prediction_tokens_ids) else None
        top3_ids_j = top3_ids[j] if j < len(top3_ids) else []
        top3_probs_j = top3_probs[j] if j < len(top3_probs) else []
        p1, p2, p3 = (top3_probs_j + [0, 0, 0])[:3]
        entropy_j = entropies[j] if j < len(entropies) else 0.0
        cal_p1 = pred_token_probs_cal[j] if j < len(pred_token_probs_cal) else 0.0
        cal_p2 = 0.0
        cal_p3 = 0.0

        aligned.append({
            "match_type": tag,
            "gt_token": gt_tokens,
            "pred_token": pred_tokens,
            "gt_token_id": gt_token_id,
            "pred_token_id": pred_token_id,
            "prob_1": p1,
            "prob_2": p2,
            "gap_13": (p1 - p3),
            "gap_12": (p1 - p2),
            "correct": gt_tokens == pred_tokens,
            "row_index": row.name, 
            "entropy": entropy_j,
            "top3_ids": top3_ids_j, 
            "relative_prob": p2 / p1 if p1 > 0 else np.nan,
            "relative_prob_cal": cal_p1
        })

    for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    add(i, j, "equal")
            elif tag == "replace":
                # Handle replacements of different lengths
                k = min(i2 - i1, j2 - j1)
                
                # Aligned substitutions
                for off in range(k): 
                    add(i1 + off, j1 + off, "replace")
                
                # Extra insertions (seq2 has more tokens)
                for j in range(j1 + k, j2): 
                    add(-1, j, "insert")  # Use -1 for no corresponding token in seq1
                
                # Extra deletions (seq1 has more tokens)
                for i in range(i1 + k, i2): 
                    add(i, -1, "delete")  # Use -1 for no corresponding token in seq2
                    
            elif tag == "insert":
                # Pure insertions (only in seq2)
                for j in range(j1, j2): 
                    add(-1, j, "insert")  # Use -1 for no corresponding token in seq1
                    
            elif tag == "delete":
                # Pure deletions (only in seq1)
                for i in range(i1, i2): 
                    add(i, -1, "delete")  # Use -1 for no corresponding token in seq2


    return aligned



def apply_error_picking(row):
    return pick_errors(
        row["pred_tokens"], 
        row["gt_tokens"],
        row["pred_token_ids"],
        row["gt_token_ids"],
        row["top3_ids"],
        row["top3_probs"],
        row["entropies"],   
        row["pred_token_probs_cal"],
        row
    )

df["token_analysis"] = df.apply(apply_error_picking, axis=1)
token_df = pd.DataFrame([t for sublist in df["token_analysis"] for t in sublist]) # flatten to dataa frame
token_df.to_csv("results/token_df.csv", index=False)

error_tokens = token_df[~token_df["correct"]]
error_tokens.to_csv("results/error_tokens.csv", index=False)


#### precision and recall (based on relative prob and entropy and the nested version of them )
total_tokens = len(token_df)
incorrect_tokens = len(token_df[~token_df["correct"]])
print(f"Total tokens: {total_tokens}, Incorrect tokens: {incorrect_tokens}, Error rate: {incorrect_tokens/total_tokens:.4f}")


def precision_and_recall(data, metric, num_thresholds=30):
    metric_vals = data[metric].dropna()
    metric_min = metric_vals.min()
    metric_max = metric_vals.max()

    if metric_max == metric_min:
        print(f"[WARN] Metric '{metric}' has constant value {metric_min}. Skipping.")
        return pd.DataFrame()

    thresholds = np.linspace(metric_min, metric_max, num=num_thresholds)

    rows = []
    for t in thresholds:
        flagged = data[data[metric] >= t]
        not_flagged = data[data[metric] < t]

        TP = (~flagged["correct"]).sum()
        FP = (flagged["correct"]).sum()
        FN = (~not_flagged["correct"]).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
        })

    return pd.DataFrame(rows)

rel_prob_table = precision_and_recall(token_df, "relative_prob_cal", num_thresholds=30)
entropy_table = precision_and_recall(token_df, "entropy",num_thresholds=30)

# print("Relative Probability based precision and recall:")
# print(rel_prob_table)   
# print("Entropy based precision and recall:")
# print(entropy_table)

# Plotting precision-recall curves
plt.figure(figsize=(10, 6))
plt.plot(rel_prob_table["recall"], rel_prob_table["precision"], marker='o', label='Relative Probability', color='blue')
plt.plot(entropy_table["recall"], entropy_table["precision"], marker='o', label='Entropy', color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig("plots/precision_recall_curve_cal.png")


# f1 score
def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return ( 2 * precision * recall) / (precision + recall)

rel_prob_table["f1"] = rel_prob_table.apply(lambda row: f1_score(row["precision"], row["recall"]), axis=1)
entropy_table["f1"] = entropy_table.apply(lambda row: f1_score(row["precision"], row["recall"]), axis=1)
print("\n==== Relative Probability - Precision, Recall, F1 ====")
print(rel_prob_table[["threshold", "precision", "recall", "f1"]])

print("\n==== Entropy - Precision, Recall, F1 ====")
print(entropy_table[["threshold", "precision", "recall", "f1"]])

best_relprob_row = rel_prob_table.loc[rel_prob_table["f1"].idxmax()]
best_entropy_row = entropy_table.loc[entropy_table["f1"].idxmax()]

print("\n==== BEST F1 SCORE - Relative Probability ====")
print(best_relprob_row)

print("\n==== BEST F1 SCORE - Entropy ====")
print(best_entropy_row)


# # plotting f1 score curves
# plt.figure(figsize=(10, 6))
# plt.plot(rel_prob_table["threshold"], rel_prob_table["f1"], marker='o', label='Relative Probability', color='blue')
# plt.plot(entropy_table["threshold"], entropy_table["f1"], marker='o', label='Entropy', color='orange')
# plt.xlabel('Threshold')
# plt.ylabel('F1 Score')
# plt.title('F1 Score vs Threshold')
# plt.legend()
# plt.grid()
# plt.savefig("results/f1_score_curve.png")


# #### AUC
# from sklearn.metrics import roc_auc_score
# # AUC for relative probability
# # We need to handle NaN values in 'relative_prob' by filling them with a value (e.g., 0)
# token_df["relative_prob_filled"] = token_df["relative_prob"].fillna(0)
# auc_rel_prob = roc_auc_score(~token_df["correct"], token_df["relative_prob_filled"])
# auc_entropy = roc_auc_score(~token_df["correct"], token_df["entropy"])
# print(f"AUC for Relative Probability: {auc_rel_prob:.4f}")
# print(f"AUC for Entropy: {auc_entropy:.4f}")

# # roc
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(~token_df["correct"], token_df["relative_prob_filled"])
# plt.title('ROC Curve - Relative Probability')
# plt.savefig("results/roc_curve_relative_probability.png")
# RocCurveDisplay.from_predictions(~token_df["correct"], token_df["entropy"])
# plt.title('ROC Curve - Entropy')
# plt.savefig("results/roc_curve_entropy.png")



########################################################################
########################################################################
metrics = {
    "entropy": 0.108058,
    "relative_prob": 0.027079,
}

incorrect_mask = token_df["gt_token_id"] != token_df["pred_token_id"]
recoverable_mask = token_df.apply(lambda row: row["gt_token_id"] in row["top3_ids"], axis=1)

for metric, threshold in metrics.items():
    # Avoid NaNs and define flagged tokens
    metric_vals = token_df[metric].fillna(0)
    flagged_mask = metric_vals > threshold

    # Combine masks
    flagged_tokens = token_df[flagged_mask]
    incorrect_tokens = token_df[incorrect_mask]
    flagged_incorrect_tokens = token_df[flagged_mask & incorrect_mask]
    recoverable_incorrect_tokens = token_df[incorrect_mask & recoverable_mask]
    recoverable_flagged_incorrect_tokens = token_df[flagged_mask & incorrect_mask & recoverable_mask]

    print(f"\n=== {metric.upper()} ===")
    print(f"Threshold: {threshold:.6f}")
    print(f"Total flagged tokens: {flagged_mask.sum()}")
    print(f"Incorrect tokens: {incorrect_mask.sum()}")
    print(f"Recoverable (Top-3 among incorrect): {recoverable_incorrect_tokens.shape[0]}")
    print(f"Flagged & incorrect: {flagged_incorrect_tokens.shape[0]}")
    print(f"Recoverable & flagged: {recoverable_flagged_incorrect_tokens.shape[0]}")

    recovery_rate = recoverable_incorrect_tokens.shape[0] / incorrect_mask.sum()
    print(f"Top-3 Recovery Rate: {recovery_rate:.4f}")

    if flagged_incorrect_tokens.shape[0] > 0:
        flagged_recovery_rate = recoverable_flagged_incorrect_tokens.shape[0] / flagged_incorrect_tokens.shape[0]
        print(f"Top-3 Recovery Rate (Flagged Incorrect): {flagged_recovery_rate:.4f}")
    else:
        print("No flagged incorrect tokens.")

# Save recoverable token pairs
recoverable_tokens = token_df[
    (~token_df["correct"]) &
    (recoverable_mask)
]

confusion_pairs = list(zip(
    recoverable_tokens["gt_token"].astype(str), 
    recoverable_tokens["pred_token"].astype(str)
))
confusion_counter = Counter(confusion_pairs)
confusion_df = pd.DataFrame(confusion_counter.items(), columns=["(gt_token, pred_token)", "count"])
confusion_df = confusion_df.sort_values(by="count", ascending=False)
confusion_df.to_csv("results/recoverable_confusions.csv", index=False)
print(f"Top recoverable confusions saved to results/recoverable_confusions.csv")
print(confusion_df.head(10))


###########################
# calibration