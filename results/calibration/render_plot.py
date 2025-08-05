import json

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind",
)
palette = sns.color_palette("colorblind")
green  = palette[2]
purple = palette[4]


def select_threshold(
    probs_val,
    labels_val,
    mode="max_f1",
    recall_target=0.80
):
    prec, rec, thr = precision_recall_curve(labels_val, probs_val)
    thr = np.append(thr, 1.0)

    if mode == "max_f1":
        f1  = 2 * prec * rec / (prec + rec + 1e-9)
        idx = np.argmax(f1)
    elif mode == "min_fp":
        mask = rec >= recall_target
        if not mask.any():
            raise ValueError("No threshold meets recall target")
        # choose threshold giving highest precision among those masks
        idx = np.argmax(prec[mask])
        idx = np.arange(len(mask))[mask][idx]
    else:
        raise ValueError("mode must be 'max_f1' or 'min_fp'")

    tau = thr[idx]
    yhat = (probs_val >= tau).astype(int)
    fp = np.sum((yhat == 1) & (np.array(labels_val) == 0))

    stats = dict(
        precision = prec[idx],
        recall = rec[idx],
        f1 = f1_score(labels_val, yhat),
        fp = int(fp)
    )
    return tau, stats


def threshold_plot(probs, labels, fp_budget=43, ratio="1:1", ax=None):
    prec, rec, thr = precision_recall_curve(labels, probs)
    thr = np.append(thr, 1.0)

    # compute FP for each threshold
    fp = []
    for idx in range(len(thr)):
        tau = thr[idx]
        yhat = (probs >= tau).astype(int)
        fp.append(np.sum((yhat == 1) & (np.array(labels) == 0)).item())
    fp = np.array(fp)

    # left axis: precision & recall
    ax.plot(thr, prec * 100, label="Precision", lw=2.5, color=green)
    ax.plot(thr, rec * 100, label="Recall", lw=2.5, ls="--", color=purple)
    ax.set_xlabel("Threshold", fontsize=13)
    ax.set_ylabel("Precision / Recall (%)", fontsize=13, labelpad=0)
    ax.set_title(f"{ratio} Vulnerable:Safe ratio", fontsize=13)

    # right axis: FP count
    ax2 = ax.twinx()
    ax2.plot(thr, fp, label="FP count", color="gray", ls=":", lw=2.5)
    ax2.set_ylabel("False positives", fontsize=13)

    # shaded acceptable region
    good = np.array(fp) <= fp_budget
    if good.any():
        tau_min, tau_max = thr[good][-1], thr[good][0]
        ax.axvspan(tau_min, tau_max, color="green", alpha=0.12,)
        ax.text(
            (tau_max + tau_min) / 2,
            50,
            f"FP$\leq${fp_budget}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="green",
            alpha=0.8,
        )

    ax.set_xlim(min(thr), 1.0)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, fp.max() * 1.15)
    ax2.tick_params(labelsize=12)
    ax2.grid(False, axis="both")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.0,
        bottom=True,
        top=False,
        labelsize=12
    )
    ax.grid(False, axis="both")
    sns.despine(ax=ax, right=False)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    return prec, rec, fp, thr, lines + lines2, labels + labels2


with open("test_python_t0.2_r1:1_s42_c0.7_metrics.json") as f:
    data = json.load(f)
    labels = data["labels"]
    scores = data["scores"]
    fp_budget = 43

# tau, val_stats = select_threshold(scores, labels, mode="min_fp", recall_target=0.75)
# print(f"Chosen τ = {tau:.4f}", val_stats)
fig = plt.figure(figsize=(6.6, 3.5))      # wide enough for two panels
gs  = fig.add_gridspec(1, 2)  # last row shorter

ax1 = fig.add_subplot(gs[0:2, 0])
p, r, fp, thr, legend_handles, legend_labels = threshold_plot(scores, labels, fp_budget=fp_budget, ratio="1:1", ax=ax1)

with open("test_python_t0.2_r1:10_s42_c0.7_metrics.json") as f:
    data = json.load(f)
    labels = data["labels"]
    scores = data["scores"]
    fp_budget = 591

ax2 = fig.add_subplot(gs[0:2, 1])
p, r, fp, thr, _, _ = threshold_plot(scores, labels, fp_budget=fp_budget, ratio="1:10", ax=ax2)

# Shared legend above both subplots
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
    fontsize=14
)

def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)

"""
print(f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>10} {'Threshold':>10}")
for prec, rec, f, thres in zip(p, r, fp, thr):
    print(f"{prec:10.4f} {rec:10.4f} {compute_f1_score(prec, rec):10.4f} {f} {thres:10.4f}")
"""

fig.tight_layout(pad=0.5)
fig.subplots_adjust(top=0.85)
fig.savefig("calibration.pdf", dpi=300, bbox_inches="tight")
plt.show()