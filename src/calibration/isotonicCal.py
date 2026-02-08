import pandas as pd
import ast
import numpy as np
from sklearn.isotonic import IsotonicRegression

CSV_PATH = "data/predictions/predictions_with_logits_normalized.csv"
OUT_PATH = "data/predictions/predictions_with_calibrated_probs.csv"
SEED = 0

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else []

# =============================
# LOAD + FLATTEN TOKENS
# =============================
df = pd.read_csv(CSV_PATH)

token_conf = []
token_correct = []
token_line_idx = []

for line_i, row in df.iterrows():
    gt_ids = parse_list(row["gt_token_ids"])
    pred_ids = parse_list(row["pred_token_ids"])
    probs = parse_list(row["pred_token_probs"])

    n = min(len(gt_ids), len(pred_ids), len(probs))
    for j in range(n):
        token_conf.append(float(probs[j]))
        token_correct.append(int(pred_ids[j] == gt_ids[j]))
        token_line_idx.append(line_i)

token_conf = np.clip(np.array(token_conf), 1e-6, 1 - 1e-6)
token_correct = np.array(token_correct)
token_line_idx = np.array(token_line_idx)

# =============================
# SPLIT BY LINE (NO LEAKAGE)
# =============================
rng = np.random.default_rng(SEED)
lines = np.unique(token_line_idx)
rng.shuffle(lines)

split = int(0.75 * len(lines))
cal_lines = set(lines[:split])

cal_mask = np.array([i in cal_lines for i in token_line_idx])

p_cal = token_conf[cal_mask]
y_cal = token_correct[cal_mask]

# =============================
# FIT ISOTONIC CALIBRATION
# =============================
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_cal, y_cal)

# =============================
# APPLY CALIBRATION PER LINE
# =============================
calibrated_probs_col = []

for _, row in df.iterrows():
    probs = parse_list(row["pred_token_probs"])
    probs = np.clip(np.array(probs, dtype=float), 1e-6, 1 - 1e-6)
    probs_cal = iso.predict(probs)
    calibrated_probs_col.append(probs_cal.tolist())

df["pred_token_probs_cal"] = calibrated_probs_col
df.to_csv(OUT_PATH, index=False)

print(f"Saved calibrated CSV to: {OUT_PATH}")


# ==========================================================
# BINNED ACCURACY: ALL vs TEST ONLY
# ==========================================================
print("\nCalibrated prob range → empirical accuracy")
print("=" * 70)

BINS = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
        (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),
        (0.8,0.9),(0.9,1.0)]

cal_probs_all, correct_all = [], []
cal_probs_test, correct_test = [], []

for line_i, row in df.iterrows():
    probs_cal = row["pred_token_probs_cal"]
    gt_ids = parse_list(row["gt_token_ids"])
    pred_ids = parse_list(row["pred_token_ids"])

    n = min(len(probs_cal), len(gt_ids), len(pred_ids))

    for i in range(n):
        is_correct = int(gt_ids[i] == pred_ids[i])

        # all tokens
        cal_probs_all.append(probs_cal[i])
        correct_all.append(is_correct)

        # test tokens only
        if line_i not in cal_lines:
            cal_probs_test.append(probs_cal[i])
            correct_test.append(is_correct)

cal_probs_all = np.array(cal_probs_all)
correct_all = np.array(correct_all)

cal_probs_test = np.array(cal_probs_test)
correct_test = np.array(correct_test)

print("\n--- ALL TOKENS ---")
for lo, hi in BINS:
    mask = (cal_probs_all >= lo) & (cal_probs_all < hi)
    count = mask.sum()
    acc = correct_all[mask].mean() if count > 0 else float("nan")

    print(f"{int(lo*100):02d}-{int(hi*100):02d} | tokens={count:6d} | acc={acc:.4f}")

print("\n--- TEST SET ONLY ---")
for lo, hi in BINS:
    mask = (cal_probs_test >= lo) & (cal_probs_test < hi)
    count = mask.sum()
    acc = correct_test[mask].mean() if count > 0 else float("nan")

    print(f"{int(lo*100):02d}-{int(hi*100):02d} | tokens={count:6d} | acc={acc:.4f}")
