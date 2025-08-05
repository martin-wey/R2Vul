import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

LANGS = ("java",)
METHODS = ("r2vul", "sft")
RUNS = ("1", "2", "3")

pred = {meth: {lang:{} for lang in LANGS} for meth in ("r2vul","sft")}
for meth in pred:
    for run in RUNS:
        for lang in LANGS:
            path = Path(f"{meth}_{run}/test_{lang}_t0.2_r1:1_s42_metrics.json")
            pred[meth][lang][run] = np.array(json.load(open(path))["predictions"])
            y_true = {"java": np.array(json.load(open(path))["labels"])}

rows = []
for m in METHODS:
    row = {"method": m}
    f1_macro_runs = []
    for l in LANGS:
        precision_runs = [precision_score(y_true[l], pred[m][l][r]) for r in RUNS]
        recall_runs = [recall_score(y_true[l], pred[m][l][r]) for r in RUNS]
        f1_runs = [f1_score(y_true[l], pred[m][l][r]) for r in RUNS]

        row["precision"] = np.mean(precision_runs) * 100
        row["precision_std"] = np.std(precision_runs) * 100
        row["recall"] = np.mean(recall_runs) * 100
        row["recall_std"] = np.std(recall_runs) * 100
        row["f1"] = np.mean(f1_runs) * 100
        row["f1_std"] = np.std(f1_runs) * 100

    rows.append(row)
tbl = pd.DataFrame(rows)

print(tbl.to_markdown())

def paired_bootstrap(labels, preds_a, preds_b, metric_func, n_rep=10_000, alpha=0.05):
    n = len(labels)
    deltas = []
    for _ in range(n_rep):
        idx = np.random.randint(0, n, n)
        deltas.append(
            metric_func(labels[idx], preds_a[idx]) -
            metric_func(labels[idx], preds_b[idx])
        )
    lo, hi = np.percentile(deltas, [alpha/2*100, (1-alpha/2)*100])
    return lo > 0

metrics = {
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

sig = {m: False for m in metrics.keys()}

y = np.tile(y_true["java"], len(RUNS))
p_r2 = np.concatenate([pred["r2vul"]["java"][r] for r in RUNS])
p_sft = np.concatenate([pred["sft"]["java"][r] for r in RUNS])

for metric, metric_func in metrics.items():
    sig[metric] = paired_bootstrap(y, p_r2, p_sft, metric_func)

print(sig)