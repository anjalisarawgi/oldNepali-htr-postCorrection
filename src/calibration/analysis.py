import pandas as pd, ast, numpy as np
import matplotlib.pyplot as plt

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else []

# # Load only the first line
df = pd.read_csv("data/predictions/predictions_with_logits_normalized.csv", nrows=5)

# gt_ids   = parse_list(df.loc[0, "gt_token_ids"])
# pred_ids = parse_list(df.loc[0, "pred_token_ids"])
# probs    = parse_list(df.loc[0, "pred_token_probs"])

# n = min(len(gt_ids), len(pred_ids), len(probs))

# conf = np.array(probs[:n], dtype=float)
# correct = np.array([pred_ids[i] == gt_ids[i] for i in range(n)], dtype=int)

# # Bin confidences
# bins = np.linspace(0, 1, 11)
# bin_ids = np.digitize(conf, bins) - 1

# mean_conf, acc = [], []
# for b in range(len(bins) - 1):
#     mask = bin_ids == b
#     if mask.any():
#         mean_conf.append(conf[mask].mean())
#         acc.append(correct[mask].mean())

# # Plot
# plt.figure()
# plt.plot([0, 1], [0, 1])
# plt.plot(mean_conf, acc, marker="o")
# plt.xlabel("Mean predicted confidence")
# plt.ylabel("Empirical accuracy")
# plt.title("Token-level reliability (first line)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(True)
# plt.savefig("calibration_plot.png")

import pandas as pd, ast, numpy as np
import matplotlib.pyplot as plt

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else []

# df = pd.read_csv("pred.csv")

all_conf = []
all_correct = []

for _, row in df.iterrows():
    gt_ids   = parse_list(row["gt_token_ids"])
    pred_ids = parse_list(row["pred_token_ids"])
    probs    = parse_list(row["pred_token_probs"])

    n = min(len(gt_ids), len(pred_ids), len(probs))
    for i in range(n):
        all_conf.append(float(probs[i]))
        all_correct.append(int(pred_ids[i] == gt_ids[i]))

all_conf = np.array(all_conf)
all_correct = np.array(all_correct)

# Bin confidences
bins = np.linspace(0, 1, 11)
bin_ids = np.digitize(all_conf, bins) - 1

mean_conf, acc = [], []
for b in range(len(bins) - 1):
    mask = bin_ids == b
    if mask.any():
        mean_conf.append(all_conf[mask].mean())
        acc.append(all_correct[mask].mean())

# Plot
plt.figure()
plt.plot([0, 1], [0, 1])
plt.plot(mean_conf, acc, marker="o")
plt.xlabel("Mean predicted confidence")
plt.ylabel("Empirical accuracy")
plt.title("Token-level reliability (all lines)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("calibration_plot_all.png")