import json
import numpy as np
import pandas as pd
from pathlib import Path

from datasets import load_from_disk
from sklearn.metrics import f1_score, precision_score, recall_score

LANGS = ("c_sharp", "javascript", "java", "python", "c")
METHODS = ("r2vul_7B", "sft_7B")
RUNS = ("1", "2", "3")
METRICS = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}
METRIC_FUNC = METRICS["recall"]

# ground‑truth labels per language
test_ds = load_from_disk("../../data/r2vul_dataset")["test"]
y_true = {lang: np.array(test_ds.filter(lambda x, l=lang: x["lang"] == l)["vulnerable"]) for lang in LANGS}

pred = {meth: {lang:{} for lang in LANGS} for meth in METHODS}
for meth in pred:
    for run in RUNS:
        for lang in LANGS:
            path = Path(f"{meth}_{run}/test_{lang}_t0.2_r1:1_s42_metrics.json")
            pred[meth][lang][run] = np.array(json.load(open(path))["predictions"])

rows = []
for m in METHODS:
    row = {"method": m}
    metric_macro_runs = []
    for l in LANGS:
        metric_runs = [METRIC_FUNC(y_true[l], pred[m][l][r]) for r in RUNS]
        row[l] = np.mean(metric_runs) * 100
        row[l+"_std"] = np.std(metric_runs, ddof=1) * 100
    # macro per run, then mean / std across runs
    for r in RUNS:
        metric_macro_runs.append(np.mean([METRIC_FUNC(y_true[l], pred[m][l][r]) for l in LANGS]))
    row["macro"] = np.mean(metric_macro_runs) * 100
    row["macro_std"] = np.std(metric_macro_runs, ddof=1) * 100
    rows.append(row)

tbl = pd.DataFrame(rows)

def paired_bootstrap(labels, preds_a, preds_b, n_rep=10_000, alpha=0.05):
    n = len(labels)
    deltas = []
    for _ in range(n_rep):
        idx = np.random.randint(0, n, n)
        deltas.append(
            f1_score(labels[idx], preds_a[idx]) -
            f1_score(labels[idx], preds_b[idx])
        )
    lo, hi = np.percentile(deltas, [alpha/2*100, (1-alpha/2)*100])
    return lo > 0

sig = {l: False for l in LANGS+("macro",)}

for l in LANGS:
    # concatenate runs so sample pairing is preserved
    y = np.tile(y_true[l], len(RUNS))
    p_r2 = np.concatenate([pred[METHODS[0]][l][r] for r in RUNS])
    p_sft = np.concatenate([pred[METHODS[1]][l][r] for r in RUNS])
    sig[l] = paired_bootstrap(y, p_r2, p_sft)

# macro test
y_all, r2_all, sft_all = [], [], []
for r in RUNS:
    for l in LANGS:
        y_all .append(y_true[l])
        r2_all.append(pred[METHODS[0]][l][r])
        sft_all.append(pred[METHODS[1]][l][r])
y_all = np.concatenate(y_all)
r2_all = np.concatenate(r2_all)
sft_all = np.concatenate(sft_all)
sig["macro"] = paired_bootstrap(y_all, r2_all, sft_all)

print(sig)
