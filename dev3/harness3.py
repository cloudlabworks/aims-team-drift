"""v3 local harness: UNSEEN-SUBJECT diagonal split + kit-exact acquisition (labeling.py) + per-target detail output.

Fold f of the diagonal design = benchmarks in benchmark-fold f (dev/folds.py) x subjects in subject-group-fold f (dev3/groups.py).
Episodes are built only from those subjects; the prior used to score them must be trained without fold-f benchmarks AND without
fold-f subjects (dev3/eval3.py). Everything else follows dev/harness.py (50/50 acquisition/evaluation split per pair, shared item
pool capped at 1000 unique subject-item pairs, nested budgets 0/1/3/7/15/31, ALC weights).
The reveal decision is labeling.acquisition_function (sha256 rule, byte-identical to the kit's random mode) and the label of a
repeated pair is labeling.construct_label (hash, not RNG state), so an episode is a pure function of (benchmark, S, seed, split).
"""
import hashlib, json, math, os, pickle, sys
import numpy as np
sys.path.insert(0, 'dev'); sys.path.insert(0, 'submission3')
import harness as H, folds as FO
import labeling as LB

K = 5
EPI = 'work/episodes3'
GR = json.load(open('work/subject_groups.json'))
BFOLDS = FO.folds(K)
BFOLD = {b: i for i, f in enumerate(BFOLDS) for b in f}
BUDGETS, W = H.BUDGETS, H.W


def subject_fold(b):
    return np.array([GR['fold_of'][g] for g in GR['per_bench'][b]], dtype=np.int8)


def make_episode(b, S, seed, allowed, cap=1000, min_items=80, indep=False):
    d = H.load(b); rng = np.random.default_rng(seed)
    nS = len(d['subjects'])
    order = np.argsort(d['si'], kind='stable'); si = d['si'][order]
    starts = np.searchsorted(si, np.arange(nS + 1))
    elig = [s for s in range(nS) if starts[s + 1] - starts[s] >= min_items and allowed[s]]
    if not elig:
        return None
    chosen = list(rng.permutation(elig)[:S])
    pool_n = max(min_items, cap // len(chosen))
    first = chosen[0]; fi = d['ii'][order][starts[first]:starts[first + 1]]
    pool = set(rng.permutation(fi)[:pool_n].tolist())
    bid = 'benchmark_' + hashlib.sha1(f'{b}|{seed}'.encode()).hexdigest()[:8]
    pairs = []
    for s in chosen:
        sl = slice(starts[s], starts[s + 1]); ii = d['ii'][order][sl]; n = d['n'][order][sl]; k = d['k'][order][sl]
        inter = [d['inter'][j] for j in order[sl]] if d['inter'] is not None else None
        if indep:
            sel = [int(j) for j in rng.permutation(len(ii))[:pool_n]]
        else:
            sel = [j for j in range(len(ii)) if ii[j] in pool]
        if len(sel) < min_items:
            continue
        sel = np.array(sel); perm = rng.permutation(len(sel)); half = len(sel) // 2
        acq = sel[perm[:half]]; ev = sel[perm[half:]]
        subj = H.subj_dict(d, s); pi = len(pairs)
        got, remaining = [], 31
        for pos, j in enumerate(acq):
            if remaining <= 0:
                break
            it = H.item_dict(d, int(ii[j]), bid, inter[j] if inter else '')
            ctx = dict(subject_id=f'subject_{pi:03d}', benchmark_id=bid, labels_remaining=remaining,
                       labels_acquired=len(got), max_labels=31, items_remaining=len(acq) - pos)
            if LB.acquisition_function([dict(subj), it], context=ctx):
                got.append(j); remaining -= 1
        labels = [(int(ii[j]), (inter[j] if inter else ''),
                   LB.construct_label(int(k[j]), int(n[j]), f'{seed}|{s}|{int(ii[j])}')) for j in got]
        evl = [(int(ii[j]), (inter[j] if inter else ''), int(n[j]), int(k[j])) for j in ev]
        pairs.append(dict(subject=subj, sid=int(s), group=GR['per_bench'][b][s], labels=labels, eval=evl))
    if not pairs:
        return None
    used = {x[0] for p in pairs for x in p['labels'] + p['eval']}
    items = {i: H.item_dict(d, i, bid, '') for i in used}
    return dict(bench=b, S=S, seed=seed, bid=bid, pairs=pairs, items=items, fold=BFOLD[b])


def build_all(S, draws=1, indep=False):
    os.makedirs(EPI, exist_ok=True)
    out = []
    for b in sorted(BFOLD):
        allowed = subject_fold(b) == BFOLD[b]
        for dr in range(draws):
            seed = int(hashlib.sha1(f'v3|{b}|{S}|{dr}'.encode()).hexdigest()[:8], 16)
            e = make_episode(b, S, seed, allowed, indep=indep)
            if e:
                out.append(e)
    pickle.dump(out, open(f"{EPI}/S{S}{'i' if indep else ''}.pkl", 'wb'), protocol=4)
    return out


def run_episode_detail(ep, predict_batch):
    """predict_batch(list_of_inputs, labeled) -> floats. Returns per-target arrays over all budgets (targets repeat per budget):
    pair (index into ep['pairs']), bud (0..5), p, n, k, nl (labels held by that pair at that budget)."""
    tg = [(pi, ii, inter, nn, kk) for pi, p in enumerate(ep['pairs']) for (ii, inter, nn, kk) in p['eval']]
    out = {key: [] for key in ('pair', 'bud', 'p', 'n', 'k', 'nl')}
    for bi, nb in enumerate(BUDGETS):
        L = H.labeled_at(ep, nb)
        targets = [H.inp(ep, ep['pairs'][pi], ii, inter) for (pi, ii, inter, nn, kk) in tg]
        preds = np.asarray(predict_batch(targets, L), dtype=np.float64)
        assert preds.shape == (len(tg),) and np.all(np.isfinite(preds)) and preds.min() >= 0 and preds.max() <= 1
        out['pair'].append(np.array([t[0] for t in tg], np.int32)); out['bud'].append(np.full(len(tg), bi, np.int8))
        out['p'].append(preds.astype(np.float32)); out['n'].append(np.array([t[3] for t in tg], np.int32))
        out['k'].append(np.array([t[4] for t in tg], np.int32))
        out['nl'].append(np.array([min(nb, len(ep['pairs'][t[0]]['labels'])) for t in tg], np.int16))
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(ep, det, nbins=10):
    """Detail -> pair-level stats: brier [P,6], ece bins [P,6,nbins,3] = (sum n, sum n*p, sum k)."""
    P = len(ep['pairs'])
    p, n, k = det['p'].astype(np.float64), det['n'].astype(np.float64), det['k'].astype(np.float64)
    key = det['pair'] * 6 + det['bud']
    sq = k * (1 - p) ** 2 + (n - k) * p ** 2
    br = (np.bincount(key, sq, minlength=P * 6) / np.bincount(key, n, minlength=P * 6)).reshape(P, 6)
    b = np.minimum((p * nbins).astype(int), nbins - 1)
    bins = np.zeros((P * 6, nbins, 3))
    for c, w in enumerate((n, n * p, k)):
        np.add.at(bins[:, :, c], (key, b), w)
    return br, bins.reshape(P, 6, nbins, 3)


def alc(b):
    return (np.asarray(b) * np.array(W)).sum(-1)


def ece(bins):
    """bins [...,nbins,3] summed over whatever the caller aggregated -> ECE with 10 equal-width bins weighted by responses."""
    w, sp, sk = bins[..., 0], bins[..., 1], bins[..., 2]
    tot = w.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = np.where(w > 0, np.abs(sk - sp), 0.0).sum(-1)
    return gap / np.maximum(tot, 1)
