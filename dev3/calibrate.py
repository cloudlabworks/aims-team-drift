"""Nested calibration of the raw (uncalibrated, unclipped) v3 logits, one map per label-budget bucket.

For outer fold f the calibrator is fitted ONLY on out-of-fold predictions of fold-j episodes (j != f) made with the inner(f, j) prior
(so it never sees fold-f benchmarks or fold-f subjects, nor an in-sample prior). Families: none | temperature (z/T) | platt (a z + c)
| isotonic. The family is chosen by cross-fitting over j (fit on the other inner folds, score on the held-out one; metric = the ALC
combination of per-bucket Brier). Writes a json {fold: calib} for evaluation and, with --final, a single calibrator fitted on the outer
raw predictions of all five folds (cross-fitted by construction) for the shipped artifact.

  calibrate.py --tag NAME --S 10,3,10i --out work/calib_NAME.json [--final work/calib_final_NAME.json]
"""
import argparse, json, pickle, sys
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, 'dev3'); import harness3 as H3

W = np.array(H3.W)
EPS = 1e-6


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS); return np.log(p / (1 - p))


def gather(files):
    """Targets of all pairs; (n, k) are rescaled by 1 / (total responses of the pair at that budget) so every pair-budget weighs 1,
    matching the pair-equal ALC (heavy pairs with millions of repeated responses would otherwise dominate the fit)."""
    z, n, k, b = [], [], [], []
    for fn in files:
        for r in pickle.load(open(fn, 'rb')):
            d = r['det']; key = d['pair'] * 6 + d['bud']
            tot = np.bincount(key, d['n'].astype(float)); s = 1.0 / tot[key]
            z.append(logit(d['p'])); n.append(d['n'] * s); k.append(d['k'] * s); b.append(d['bud'])
    return np.concatenate(z), np.concatenate(n).astype(float), np.concatenate(k).astype(float), np.concatenate(b)


def fit_bucket(kind, z, n, k):
    if kind == 'none':
        return dict(a=1.0, c=0.0)
    if kind == 'iso':
        p = 1 / (1 + np.exp(-z)); o = np.argsort(p)
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip').fit(p[o], (k / n)[o], sample_weight=n[o])
        xs = np.unique(np.quantile(p, np.linspace(0, 1, 60), method='inverted_cdf'))
        return dict(x=[float(v) for v in xs], y=[float(v) for v in ir.predict(xs)])

    def nll(t):
        a, c = (t[0], 0.0) if kind == 'temp' else (t[0], t[1])
        s = a * z + c
        return float((n * np.logaddexp(0, s) - k * s).sum() / n.sum()) + 1e-4 * (a - 1) ** 2
    x0 = [1.0] if kind == 'temp' else [1.0, 0.0]
    t = minimize(nll, x0, method='L-BFGS-B').x
    return dict(a=float(t[0]), c=0.0 if kind == 'temp' else float(t[1]))


def fit(kind, z, n, k, b):
    return dict(kind='platt' if kind in ('none', 'temp', 'platt') else kind, fam=kind,
                buckets=[fit_bucket(kind, z[b == i], n[b == i], k[b == i]) for i in range(6)])


def apply(cal, z, b):
    out = np.zeros_like(z)
    for i in range(6):
        m = b == i; cb = cal['buckets'][i]
        if cal['kind'] == 'platt':
            out[m] = 1 / (1 + np.exp(-(cb['a'] * z[m] + cb['c'])))
        else:
            out[m] = np.interp(1 / (1 + np.exp(-z[m])), cb['x'], cb['y'])
    return np.clip(out, 0.01, 0.99)


def score(cal, z, n, k, b):
    p = apply(cal, z, b)
    br = [((k * (1 - p) ** 2 + (n - k) * p ** 2)[b == i]).sum() / n[b == i].sum() for i in range(6)]
    return float((np.array(br) * W).sum())


FAMS = ('none', 'temp', 'platt', 'iso')


def choose(parts):
    """parts: list of (z,n,k,b) held-out groups. Cross-fit the family; return (best, {fam: score})."""
    sc = {}
    for fam in FAMS:
        tot = 0.0
        for h in range(len(parts)):
            tr = [np.concatenate(x) for x in zip(*[p for i, p in enumerate(parts) if i != h])]
            tot += score(fit(fam, *tr), *parts[h])
        sc[fam] = tot / len(parts)
    return min(sc, key=sc.get), sc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--S', default='10,3,10i'); ap.add_argument('--out', required=True)
    ap.add_argument('--final'); ap.add_argument('--family', default='auto')
    a = ap.parse_args()
    Ss = a.S.split(','); out = {}; log = {}
    for f in range(H3.K):
        parts = []
        for j in range(H3.K):
            if j != f:
                parts.append(gather([f'work/res3/{a.tag}_S{S}_oof_f{f}_j{j}.pkl' for S in Ss]))
        fam, sc = choose(parts) if a.family == 'auto' else (a.family, {})
        out[str(f)] = fit(fam, *[np.concatenate(x) for x in zip(*parts)]); log[f] = (fam, {k: round(v, 5) for k, v in sc.items()})
        print('outer fold', f, 'family', fam, 'cross-fit ALC proxy', log[f][1], flush=True)
    json.dump(out, open(a.out, 'w'))
    if a.final:
        parts = [gather([f'work/res3/{a.tag}_S{S}_outer_f{f}.pkl' for S in Ss]) for f in range(H3.K)]
        fam, sc = choose(parts) if a.family == 'auto' else (a.family, {})
        cal = fit(fam, *[np.concatenate(x) for x in zip(*parts)])
        json.dump(cal, open(a.final, 'w')); print('final family', fam, sc)
