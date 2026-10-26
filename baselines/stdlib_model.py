"""Shrunken subject x benchmark mean with a pooled additive-logit prior.

Stdlib only, no files opened, no state kept between calls.

At budget 0 (empty ``labeled``) the prediction is 0.5. With labels, a ridge
logistic model  logit p = mu + a[subject] + b[benchmark]  is fitted on all
supplied labels; its output p0 acts as a prior for the target
subject-benchmark pair, and the pair mean is shrunk toward it:
(successes + K * p0) / (n + K). Output is clipped to [LO, HI].
"""

import json
import math

K = 3.0            # prior pseudo-count for the pair mean
LAMBDA = 1.0       # ridge strength on subject / benchmark offsets
SWEEPS = 25        # coordinate Newton sweeps
LO, HI = 0.03, 0.97


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _subject_key(subject):
    return json.dumps(subject, sort_keys=True, default=str)


def _benchmark_key(item):
    value = item.get("benchmark_id") if isinstance(item, dict) else None
    return value if isinstance(value, str) and value else None


def _fit_additive(rows):
    """rows: list of (subject_key, benchmark_key, y). Returns (mu, a, b)."""
    mu = 0.0
    a = {s: 0.0 for s, _, _ in rows}
    b = {g: 0.0 for _, g, _ in rows}
    by_s, by_g = {}, {}
    for idx, (s, g, _) in enumerate(rows):
        by_s.setdefault(s, []).append(idx)
        by_g.setdefault(g, []).append(idx)
    eta = [0.0] * len(rows)

    def newton(indices, current, ridge):
        grad = hess = 0.0
        for i in indices:
            p = _sigmoid(eta[i])
            grad += rows[i][2] - p
            hess += p * (1.0 - p)
        grad -= ridge * current
        hess += ridge
        return grad / hess if hess > 1e-9 else 0.0

    for _ in range(SWEEPS):
        step = newton(range(len(rows)), mu, 1e-3)
        mu += step
        for i in range(len(rows)):
            eta[i] += step
        for s, idx in by_s.items():
            step = newton(idx, a[s], LAMBDA)
            a[s] += step
            for i in idx:
                eta[i] += step
        for g, idx in by_g.items():
            step = newton(idx, b[g], LAMBDA)
            b[g] += step
            for i in idx:
                eta[i] += step
    return mu, a, b


def predict(input, labeled=None):
    subject, item = input
    if not labeled:
        return 0.5

    s_key = _subject_key(subject)
    g_key = _benchmark_key(item)
    rows = []
    for entry in labeled:
        (obs_subject, obs_item), label = entry
        if label not in (0, 1):
            continue
        rows.append((_subject_key(obs_subject), _benchmark_key(obs_item), int(label)))
    if not rows:
        return 0.5

    mu, a, b = _fit_additive(rows)
    p0 = _sigmoid(mu + a.get(s_key, 0.0) + b.get(g_key, 0.0))

    n = succ = 0
    for s, g, y in rows:
        if s == s_key and g == g_key:
            n += 1
            succ += y
    p = (succ + K * p0) / (n + K)
    return float(min(HI, max(LO, p)))
