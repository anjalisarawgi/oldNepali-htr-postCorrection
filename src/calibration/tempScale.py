import pandas as pd, ast, numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else []

def logit(p):
    return np.log(p) - np.log(1 - p)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def reliability_curve(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1

    mean_conf, acc = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            mean_conf.append(conf[mask].mean())
            acc.append(correct[mask].mean())
    return np.array(mean_conf), np.array(acc)

# ---------- 1) build token dataset ----------
csv_path = "data/predictions/predictions_with_logits_normalized.csv"  # change if needed
df = pd.read_csv(csv_path)

all_p = []
all_y = []

for _, row in df.iterrows():
    gt_ids   = parse_list(row["gt_token_ids"])
    pred_ids = parse_list(row["pred_token_ids"])
    probs    = parse_list(row["pred_token_probs"])

    n = min(len(gt_ids), len(pred_ids), len(probs))
    for i in range(n):
        p = float(probs[i])
        y = int(pred_ids[i] == gt_ids[i])
        all_p.append(p)
        all_y.append(y)

all_p = np.array(all_p, dtype=float)
all_y = np.array(all_y, dtype=int)

# clip probs to avoid inf logits
eps = 1e-6
all_p = np.clip(all_p, eps, 1 - eps)

print("Tokens:", len(all_p), " Accuracy:", all_y.mean().round(4), " Mean conf:", all_p.mean().round(4))

# ---------- 2) split into calibration + test ----------
rng = np.random.default_rng(0)
idx = np.arange(len(all_p))
rng.shuffle(idx)

split = int(0.5 * len(idx))   # 50/50 split (fine for a start)
cal_idx, test_idx = idx[:split], idx[split:]

p_cal, y_cal = all_p[cal_idx], all_y[cal_idx]
p_test, y_test = all_p[test_idx], all_y[test_idx]

# ---------- 3) fit Temperature T by minimizing binary NLL ----------
# We do a simple grid search (robust, no extra deps).
def nll_for_T(T, p, y):
    z = logit(p) / T
    p_calib = sigmoid(z)
    p_calib = np.clip(p_calib, eps, 1 - eps)
    return -np.mean(y * np.log(p_calib) + (1 - y) * np.log(1 - p_calib))

Ts = np.linspace(0.5, 10.0, 200)  # search range
nlls = np.array([nll_for_T(T, p_cal, y_cal) for T in Ts])
best_T = Ts[np.argmin(nlls)]

print("Best T:", best_T.round(4), "  Cal NLL:", nlls.min().round(6))

# apply on test set
p_test_calib = sigmoid(logit(p_test) / best_T)

# ---------- 4) plot reliability (test set) ----------
# Before
x1, y1 = reliability_curve(p_test, y_test, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1])
plt.plot(x1, y1, marker="o")
plt.xlabel("Mean predicted confidence")
plt.ylabel("Empirical accuracy")
plt.title("Token reliability BEFORE temp scaling (test split)")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True)
plt.savefig("reliability_before.png")

# After
x2, y2 = reliability_curve(p_test_calib, y_test, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1])
plt.plot(x2, y2, marker="o")
plt.xlabel("Mean predicted confidence (calibrated)")
plt.ylabel("Empirical accuracy")
plt.title(f"Token reliability AFTER temp scaling (T={best_T:.3f}) (test split)")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(True)
plt.savefig("reliability_after.png")