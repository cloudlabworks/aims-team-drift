"""SYNTHETIC-ONLY Brier ALC comparison (not real competition data, not a real score).

Generates Rasch-style responses, mimics the 50/50 acquisition/evaluation split and
nested label budgets, and compares: constant 0.5, organiser empirical mean, our model.

    python -I tests/synthetic_alc.py submission/model.py paiec_baseline/empirical_mean/model.py
"""
import importlib.util
import math
import random
import sys

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.predict


def sig(z):
    return 1 / (1 + math.exp(-z))


def make_world(seed, n_sub=4, n_bench=5, n_items=120):
    rng = random.Random(seed)
    subjects = [{"normalized_name": f"m{i}", "harness": "h"} for i in range(n_sub)]
    theta = [rng.gauss(0, 1.2) for _ in subjects]
    bench = [f"benchmark_{j}" for j in range(n_bench)]
    bdiff = [rng.gauss(0, 1.0) for _ in bench]
    pairs = {}
    for i, s in enumerate(subjects):
        for j, b in enumerate(bench):
            items = []
            for k in range(n_items):
                d = rng.gauss(0, 1.0)
                p = sig(theta[i] - bdiff[j] - d)
                items.append(({"item_content": f"{b}-{k}", "benchmark_id": b}, int(rng.random() < p)))
            rng.shuffle(items)
            half = len(items) // 2
            pairs[(i, j)] = (items[:half], items[half:])  # acquisition, evaluation
    return subjects, bench, pairs


def run(seed, predictors):
    subjects, bench, pairs = make_world(seed)
    out = {name: [] for name in predictors}
    for (i, j), (acq, ev) in pairs.items():
        briers = {name: [] for name in predictors}
        for n in BUDGETS:
            # evidence: first n acquisition labels of this pair + n of each other pair (shared trajectory)
            labeled = []
            for (i2, j2), (acq2, _) in pairs.items():
                for item, y in acq2[:n]:
                    labeled.append([[subjects[i2], item], y])
            for name, f in predictors.items():
                se = 0.0
                for item, y in ev[:20]:
                    se += (f([subjects[i], item], labeled) - y) ** 2
                briers[name].append(se / 20)
        for name in predictors:
            out[name].append(briers[name])
    res = {}
    for name, rows in out.items():
        per_budget = [sum(r[k] for r in rows) / len(rows) for k in range(6)]
        res[name] = (sum(w * b for w, b in zip(WEIGHTS, per_budget)), per_budget)
    return res


if __name__ == "__main__":
    ours = load(sys.argv[1], "ours")
    mean = load(sys.argv[2], "mean")
    preds = {"const_0.5": lambda inp, lab=None: 0.5, "empirical_mean": mean, "shrunk_additive": ours}
    agg = {}
    seeds = range(2)
    for s in seeds:
        for name, (alc, pb) in run(s, preds).items():
            agg.setdefault(name, []).append((alc, pb))
    for name, v in agg.items():
        alc = sum(a for a, _ in v) / len(v)
        pb = [sum(p[k] for _, p in v) / len(v) for k in range(6)]
        print(f"{name:16s} ALC={alc:.4f}  per-budget Brier={[round(x,4) for x in pb]}")
