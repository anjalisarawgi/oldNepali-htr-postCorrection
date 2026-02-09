import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

CSV_PATH = "results/lemma_matches_with_dict_matches_unmatched_ngram_matches_with_logits.csv"
OUT_DIR  = "plots/calibration"
SEED = 0
N_BINS = 15

np.random.seed(SEED)

def parse(x):
    return ast.literal_eval(x) if isinstance(x, str) else []

df = pd.read_csv(CSV_PATH)

rows = []

for line_i, row in df.iterrows():
    gt = parse(row["gt_token_ids"])
    pred = parse(row["pred_token_ids"])
    probs = parse(row["pred_token_probs"])
    logits = parse(row["pred_token_logits"]) if "pred_token_logits" in row else None

    n = min(len(gt), len(pred), len(probs))
    for i in range(n):
        rows.append({
            "line": line_i,
            "p": np.clip(float(probs[i]), 1e-6, 1 - 1e-6),
            "logit": float(logits[i]) if logits is not None else None,
            "y": int(gt[i] == pred[i])
        })

tok = pd.DataFrame(rows)

lines = tok["line"].unique()
np.random.shuffle(lines)

split = int(0.5 * len(lines))
cal_lines = set(lines[:split])

cal = tok[tok.line.isin(cal_lines)]
test = tok[~tok.line.isin(cal_lines)]

# ======================================================
# UTILS
# ======================================================
def reliability_curve(p, y, bins=15):
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(p, qs)
    acc, conf = [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc.append(y[mask].mean())
        conf.append(p[mask].mean())

    return np.array(conf), np.array(acc)

def plot_curve(before, after, title, path):
    plt.figure(figsize=(5,5))
    plt.plot(before[0], before[1], "o-", label="Before")
    plt.plot(after[0], after[1], "o-", label="After")
    plt.plot([0,1],[0,1],"k--",alpha=0.5)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ======================================================
# 1) TEMPERATURE SCALING
# ======================================================
def fit_temperature(logits, y):
    def nll(T):
        p = 1 / (1 + np.exp(-logits / T))
        return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

    res = minimize(nll, x0=[1.0], bounds=[(0.05, 10)])
    return res.x[0]

# ======================================================
# 2) PLATT SCALING
# ======================================================
platt = LogisticRegression(solver="lbfgs")
platt.fit(cal[["p"]], cal["y"])

# ======================================================
# 3) BETA CALIBRATION
# ======================================================
X_beta = np.column_stack([
    np.log(cal.p),
    np.log(1 - cal.p)
])

beta = LogisticRegression(solver="lbfgs")
beta.fit(X_beta, cal.y)

# ======================================================
# 4) ISOTONIC (PRE-BINNED)
# ======================================================
qs = np.linspace(0, 1, N_BINS + 1)
edges = np.quantile(cal.p, qs)

bin_p, bin_y, bin_w = [], [], []

for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (cal.p >= lo) & (cal.p <= hi)
    if mask.sum() == 0:
        continue
    bin_p.append(cal.p[mask].mean())
    bin_y.append(cal.y[mask].mean())
    bin_w.append(mask.sum())

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(bin_p, bin_y, sample_weight=bin_w)

# ======================================================
# APPLY + SAVE
# ======================================================
methods = {}

# baseline
methods["uncalibrated"] = tok.p.values

# temperature
if tok.logit.notna().all():
    T = fit_temperature(cal.logit.values, cal.y.values)
    methods["temperature"] = 1 / (1 + np.exp(-tok.logit.values / T))

# platt
methods["platt"] = platt.predict_proba(tok[["p"]])[:,1]

# beta
methods["beta"] = beta.predict_proba(
    np.column_stack([np.log(tok.p), np.log(1 - tok.p)])
)[:,1]

# isotonic
methods["isotonic"] = iso.predict(tok.p)

# ======================================================
# PLOTS + CSVs
# ======================================================
for name, p_cal in methods.items():
    tok[f"p_{name}"] = p_cal

    before = reliability_curve(test.p.values, test.y.values)
    after  = reliability_curve(
        tok.loc[test.index, f"p_{name}"].values,
        test.y.values
    )

    plot_curve(
        before, after,
        title=f"{name} calibration",
        path=f"{OUT_DIR}/{name}.png"
    )

tok.to_csv(f"{OUT_DIR}/tokens_with_all_calibrations.csv", index=False)

print("Saved:")
print("- per-method reliability PNGs")
print("- unified calibrated token CSV")


# ======================================================
# SAVE BETA CALIBRATION BACK TO ORIGINAL CSV
# ======================================================

# attach beta-calibrated probs to token table
tok["cal_prob_beta"] = methods["beta"]

# regroup per line, preserving token order
beta_per_line = (
    tok.sort_values(["line"])
       .groupby("line")["cal_prob_beta"]
       .apply(list)
)

# load original CSV again
df_out = pd.read_csv(CSV_PATH)

# insert new column (lists serialized like existing cols)
df_out["cal_prob_beta"] = df_out.index.map(
    lambda i: beta_per_line[i] if i in beta_per_line else []
)
OUTPUT_CSV = CSV_PATH.replace(".csv", "_with_beta_cal.csv")
df_out.to_csv(OUTPUT_CSV, index=False)
print("beta-calibrated probabilities written back as cal_prob_beta")