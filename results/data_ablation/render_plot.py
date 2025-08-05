import json
import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from matplotlib.patches import Rectangle
from sklearn.metrics import f1_score


METHODS = ("qwen2.5-1.5B-sft", "r2vul-1.5B")
FRACTIONS = ("0.25", "0.50", "0.75", "1.0")
SEEDS = (42, 404, 12345)
LANGS = ("c_sharp", "javascript", "java", "python", "c")
COMPUTE_PAIRED_BOOTSTRAP = False


records = [] # per‑seed macro-F1 scores
for method in METHODS:
    for frac in FRACTIONS:
        seed_macros = []
        for seed in SEEDS:
            per_lang = []
            for lang in LANGS:
                f = pathlib.Path(f"{method}_f{frac}_s{seed}/"
                                 f"test_{lang}_t0.2_r1:1_s42_metrics.json")
                with f.open() as fh:
                    per_lang.append(json.load(fh)["f1"] * 100)
            macro_f1 = np.mean(per_lang)
            seed_macros.append(macro_f1)
            records.append(dict(method=method, fraction=frac, seed=seed, macro_f1=macro_f1))

df_points = pd.DataFrame(records)
df_points["fraction"] = df_points["fraction"].astype(float)

offset = {"qwen2.5-1.5B-sft": -0.035, "r2vul-1.5B": 0.035}
df_points["x_pos"] = df_points.apply(
    lambda r: r["fraction"] + offset[r["method"]], axis=1
)

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind",
)
palette = sns.color_palette("colorblind")

fig = plt.figure(figsize=(5, 3))
gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
ax = fig.add_subplot(gs[0])
legend_ax = fig.add_subplot(gs[1])

seed_palette = {42: palette[2], 404: palette[7], 12345: palette[9]}   # blue, green, red
marker_dict  = {"qwen2.5-1.5B-sft": "o", "r2vul-1.5B": "X"}

sns.scatterplot(
    data=df_points,
    x="x_pos",
    y="macro_f1",
    hue="seed",
    style="method",
    palette=seed_palette,
    markers=marker_dict,
    s=70,
    ax=ax,
    legend=False,
    alpha=0.7,
)

method_color = {"qwen2.5-1.5B-sft": "black", "r2vul-1.5B": palette[1]}
for (method, frac), grp in df_points.groupby(["method", "fraction"]):
    # horizontal positions include the dodge offset
    xs = grp["fraction"] + offset[method]
    ys = grp["macro_f1"]
    # bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    pad = 0.5
    rect = Rectangle(
        (x_min - 0.005, y_min - pad), # lower‑left
        (x_max - x_min) + 0.01, # width
        (y_max - y_min) + 2 * pad, # height
        linewidth=1.2,
        edgecolor="black",
        facecolor="black",
        alpha=0.075,
        zorder=0
    )
    ax.add_patch(rect)

    # mean label at box centre
    mean_val = ys.mean()
    if method == "r2vul-1.5B" and frac in [0.50, 0.75, 1.00]:
        annotation_text = f"$\\mathbf{{{mean_val:.1f}}}^\dagger$"
    else:
        annotation_text = f"${mean_val:.1f}$"

    text_x_pos = (x_min + x_max) / 2
    text_y_pos = y_max + 1.5

    if method == "r2vul-1.5B" and frac == 0.25:
        text_x_pos += 0.03

    ax.text(
        text_x_pos,
        text_y_pos,
        annotation_text,
        color=method_color[method],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        alpha=0.8,
    )

# two legends: one for seeds (colour), one for method means
seed_handles = [
    plt.Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        markerfacecolor=seed_palette[s],
        markersize=7,
        label=f"{s}",
        alpha=.7
    )
    for s in seed_palette
]

seed_legend = legend_ax.legend(
    handles=seed_handles,
    title="Random seed",
    frameon=False,
    loc="upper left",
    fontsize=10,
    title_fontsize=11,
    borderpad=0.2,
    labelspacing=0.3,
    bbox_to_anchor=(-0.5, 1.05),
)

method_handles = [
    mlines.Line2D(
        [],
        [], marker='o',
        color='black',
        markersize=7,
        linestyle='None',
        alpha=0.7,
        label="SFT"
    ),
    mlines.Line2D(
        [],
        [],
        marker='X',
        color='black',
        markersize=7,
        linestyle='None',
        alpha=0.7,
        label="R2Vul"
    )
]

legend_ax.legend(
    handles=method_handles,
    title="Method",
    frameon=False,
    loc="lower left",
    fontsize=10,
    title_fontsize=11,
    borderpad=0.2,
    labelspacing=0.3,
    bbox_to_anchor=(-.25, .390),
    labelcolor=["black", palette[1]],
)

legend_ax.axis('off')
legend_ax.add_artist(seed_legend)

shades = [0.25, 0.5, 0.75, 1.0]
band_width = 0.25
for i, xpos in enumerate(shades):
    if i % 2 == 0:
        ax.axvspan(
            xpos - band_width / 2,
            xpos + band_width / 2,
            color='gray',
            alpha=0.05,
            zorder=0
        )

ax.set_xlim(0.1, 1.15)
ax.set_xlabel("Training data fraction", fontsize=12)
ax.set_ylabel("Macro-F$_1$ (%)", fontsize=12)
ax.set_xticks([0.25, 0.50, 0.75, 1.00])
ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=5,
    width=1.0,
    bottom=True,
    top=False,
    labelsize=11,
    left=True
)
ax.set_ylim(60, 85)
ax.grid(False, axis='both')
ax.yaxis.grid(True, linestyle='-.', alpha=0.7)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig("data_ablation.pdf", dpi=300)
plt.show()

if COMPUTE_PAIRED_BOOTSTRAP:
    def paired_bootstrap(labels, preds_a, preds_b, n_rep=10_000, alpha=0.05):
        labels = np.array(labels)
        preds_a = np.array(preds_a)
        preds_b = np.array(preds_b)

        n = len(labels)
        deltas = []
        for _ in range(n_rep):
            idx = np.random.randint(0, n, n)
            delta = f1_score(labels[idx], preds_a[idx]) - f1_score(labels[idx], preds_b[idx])
            deltas.append(delta)

        lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return lo > 0  # True if R2Vul significantly better than SFT


    # Compute significance for each fraction
    test_ds = load_from_disk("../../data/r2vul_dataset")["test"]
    y_true = {lang: np.array(test_ds.filter(lambda x, l=lang: x["lang"] == l)["vulnerable"]) for lang in LANGS}

    pred = {
        meth: {
            frac: {
                lang: {} for lang in LANGS
            } for frac in FRACTIONS
        } for meth in METHODS
    }

    for meth in pred:
        for frac in FRACTIONS:
            for seed in SEEDS:
                scores = []
                for lang in LANGS:
                    path = pathlib.Path(f"{meth}_f{frac}_s{seed}/test_{lang}_t0.2_r1:1_s42_metrics.json")
                    pred[meth][frac][lang][seed] = np.array(json.load(open(path))["predictions"])

    sig_dict = {}

    for frac in FRACTIONS:
        all_labels = []
        all_preds_r2 = []
        all_preds_sft = []

        for lang in LANGS:
            y = y_true[lang]
            for seed in SEEDS:
                preds_r2  = pred[METHODS[1]][frac][lang][seed]  # R2Vul
                preds_sft = pred[METHODS[0]][frac][lang][seed]  # SFT

                # Sanity check: length must match
                assert len(preds_r2) == len(preds_sft) == len(y), \
                    f"Length mismatch in {lang}, seed {seed}, frac {frac}"

                all_labels.append(y)
                all_preds_r2.append(preds_r2)
                all_preds_sft.append(preds_sft)

        y_concat = np.concatenate(all_labels)
        p_r2_concat = np.concatenate(all_preds_r2)
        p_sft_concat = np.concatenate(all_preds_sft)

        sig_dict[frac] = paired_bootstrap(y_concat, p_r2_concat, p_sft_concat)

    print(sig_dict)
