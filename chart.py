import matplotlib.pyplot as plt

# Data: error counts per span length bin
bins = ["3–4", "5–6", "7–8", "9≥"]
errors = [8, 13, 8, 3]

plt.figure(figsize=(6, 4))
bars = plt.bar(bins, errors, color='salmon')

# Add value labels on top of bars
for bar, err in zip(bars, errors):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, str(err),
             ha='center', va='bottom', fontsize=10)

plt.title("Error Counts by Matched Word Length")
plt.xlabel("Length of the matched words")
plt.ylabel("Number of wrong characters")
plt.ylim(0, max(errors) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("error_counts_by_span_length.png")