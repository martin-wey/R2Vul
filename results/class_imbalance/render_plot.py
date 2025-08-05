import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASET_DIR = "../../data/r2vul_dataset"
METRIC_GLOB = "test_java_t0.2_r*_s42_metrics.json"
RATIOS = ("1:1","1:2","1:3","1:4","1:5","1:10","1:20")
N_BOOT = 10_000

def fp_ci(labels, preds, n_boot=10_000, alpha=0.05):
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        fp = np.sum((preds[idx] == 1) & (labels[idx] == 0))
        diffs.append(fp)
    lo, hi = np.percentile(diffs, [alpha/2*100, (1-alpha/2)*100])
    return int(np.mean(diffs)), int(lo), int(hi)

records = []
# 3. iterate over ratios
for ratio in RATIOS:
    for method in {"r2vul", "sft"}:
        fpath = Path(os.path.join(method, Path(METRIC_GLOB.replace("*", ratio))))
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} not found")

        preds = np.array(json.load(open(fpath))["predictions"])
        y_true = np.array(json.load(open(fpath))["labels"])
        fp_cnt = np.sum((preds == 1) & (y_true == 0))
        fp_mean, fp_lo, fp_hi = fp_ci(y_true, preds, N_BOOT)

        records.append({
            "method": method,
            "ratio": ratio,
            "fp": fp_cnt.item(),
            "ci_lo": fp_lo,
            "ci_hi": fp_hi,
            "n": np.sum(y_true == 0),
        })


df = pd.DataFrame(records)
df["fpr"] = df["fp"] / df["n"] * 100
df["fpr_lo"] = df["ci_lo"] / df["n"] * 100
df["fpr_hi"] = df["ci_hi"] / df["n"] * 100

df_sft = df[df["method"] == "sft"]
df_r2vul = df[df["method"] == "r2vul"]

df_sft["x"] = df_sft["ratio"].str.split(":").str[1].astype(int)
df_r2vul["x"] = df_r2vul["ratio"].str.split(":").str[1].astype(int)

plot_df = pd.concat([
    df_sft.assign(Method="SFT"),
    df_r2vul.assign(Method="R2Vul"),
])

sns.set_theme(
    style="whitegrid",          # light grid; no top/right spines
    context="paper",            # font sizes tuned for a PDF
    palette="colorblind",       # safe 8‑colour palette
)
fig, ax = plt.subplots(figsize=(5, 3))

sns.lineplot(
    data=plot_df,
    x="x", y="fpr",
    hue="Method",
    style="Method",
    markers=True,
    dashes=False,
    linewidth=2.2,
    markersize=8,
    ax=ax,
)

for _, grp in plot_df.groupby("ratio"):
    if len(grp) == 2:
        x = grp["x"].iloc[0]
        y_vals = grp.sort_values("Method")["fpr"].values
        fp_vals = grp.sort_values("Method")["fp"].values
        ax.plot([x, x], y_vals, lw=1, ls=":", color="gray", alpha=.6)
        ax.text(x*0.96, np.mean(y_vals)*0.9535, f"Δ={fp_vals[0]-fp_vals[1]}",
                ha="center", va="bottom", rotation=90, fontsize=9, color="gray", fontweight="bold")

for ratio in RATIOS:
    pt = df_sft[df_sft["ratio"] == ratio].iloc[0]
    x_ratio = 0.95 if ratio == "1:5" else 0.88
    y_ratio = 1.025 if ratio in {"1:1", "1:2", "1:3"} else 1.020

    ax.annotate(
        f'n={int(pt.fp)}',
        xy=(pt.x, pt.fpr),
        xytext=(pt.x*x_ratio, pt.fpr*y_ratio),
        fontsize=9
    )

    pt = df_r2vul[df_r2vul["ratio"] == ratio].iloc[0]
    x_ratio = 0.90
    y_ratio = 0.945 if ratio in {"1:1", "1:2", "1:3"} else 0.955
    ax.annotate(
        f'n={int(pt.fp)}',
        xy=(pt.x, pt.fpr),
        xytext=(pt.x * x_ratio, pt.fpr * y_ratio),
        fontsize=9,
    )

ax.set_xscale("log")
ax.set_xticks(df_sft["x"])
ax.set_xticklabels(df_sft["ratio"], fontsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel("Vulnerable:Safe ratio", fontsize=12)
ax.set_ylabel("False positive rate (%)", fontsize=12)
ax.set_ylim(19, max(plot_df["fpr"]) + 3)
ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=5,
    width=1.0,
    bottom=True,
    top=False
)
ax.tick_params(
    axis="y",
    which="major",
    direction="out",
    length=5,
    width=1.0,
    left=True,
    right=False
)
ax.grid(False, axis="both")
ax.yaxis.grid(True, linestyle='-.', alpha=0.7)
ax.legend(frameon=False, fontsize=11)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig("class_imbalance_java.pdf", dpi=300)
plt.show()
