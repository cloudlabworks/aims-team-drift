"""Pair-level tables + grouped (cluster) bootstrap for the unseen-subject split.

  analyze.py --S 10 --models const05:NAME empirical_mean:NAME v2:NAME v3:NAME ... [--ref v2] [--kit empirical_mean] [--B 4000]
Each model spec is label:tag (files work/res3/{tag}_S{S}_outer_f{f}.pkl). Everything is pair-level (pairs weigh equally, as in the
organisers' ALC); the bootstrap resamples whole CLUSTERS with replacement: benchmarks (primary - the test is a benchmark cold
start) and, as a second view, subject groups (model identities). Paired deltas use the same resampled clusters for both models.
ECE = 10 equal-width bins; computed PER PAIR and per budget (responses weighted), then averaged over pairs (equal weight, like ALC)
and over the six budgets; also shown at B0 and B31 and, for reference, pooled over all responses (dominated by a few pairs with
millions of repeated responses). The organisers' exact ECE pooling is unknown (diagnostic only).
"""
import argparse, pickle, sys
import numpy as np
sys.path.insert(0, 'dev3')
import harness3 as H3


def load(tag, S):
    rows = []
    for f in range(H3.K):
        rows += pickle.load(open(f'work/res3/{tag}_S{S}_outer_f{f}.pkl', 'rb'))
    bench, group, br, bins = [], [], [], []
    for r in rows:
        for gi, g in enumerate(r['groups']):
            bench.append(r['bench']); group.append(g)
        br.append(r['br']); bins.append(r['bins'])
    return np.array(bench), np.array(group), np.concatenate(br), np.concatenate(bins)


def cluster_stats(keys, br, bins):
    u, inv = np.unique(keys, return_inverse=True)
    C = len(u)
    n = np.bincount(inv, minlength=C).astype(float)
    sb = np.zeros((C, 6)); np.add.at(sb, inv, br)
    sbins = np.zeros((C,) + bins.shape[1:]); np.add.at(sbins, inv, bins)
    se = np.zeros((C, 6)); np.add.at(se, inv, H3.ece(bins))  # per-pair, per-budget ECE summed per cluster
    return n, sb, sbins, se


def metrics(n, sb, sbins, se, w=None):
    """w: cluster weights [R,C] (or None = the sample itself). Returns dict of arrays with leading dim R."""
    if w is None:
        w = np.ones((1, len(n)))
    N = w @ n
    B = (w @ sb) / N[:, None]
    bins = np.tensordot(w, sbins, axes=(1, 0))  # [R,6,10,3]
    E = (w @ se) / N[:, None]  # pair-mean ECE per budget
    return dict(B=B, ALC=(B * np.array(H3.W)).sum(1), ECE=E.mean(1), ECE0=E[:, 0], ECE31=E[:, 5], ECEpool=H3.ece(bins.sum(1)))


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, lo), np.percentile(x, hi)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--S', default='10'); ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--ref', default='v2'); ap.add_argument('--kit', default='empirical_mean'); ap.add_argument('--B', type=int, default=4000)
    ap.add_argument('--clusters', default='bench,group')
    a = ap.parse_args()
    data = {}
    for spec in a.models:
        lab, tag = spec.split(':'); data[lab] = load(tag, a.S)
    labs = list(data); b0, g0 = data[labs[0]][0], data[labs[0]][1]
    for l in labs:
        assert (data[l][0] == b0).all() and (data[l][1] == g0).all(), f'pair mismatch for {l}'
    print(f'### S={a.S}: {len(b0)} pairs, {len(set(b0))} benchmarks, {len(set(g0))} subject groups (unseen-subject diagonal split)')
    rng = np.random.default_rng(12345)
    for ckey in a.clusters.split(','):
        keys = b0 if ckey == 'bench' else g0
        C = len(set(keys)); W = rng.multinomial(C, np.ones(C) / C, size=a.B).astype(float)
        M, M0 = {}, {}
        for l in labs:
            n, sb, sbins, se = cluster_stats(keys, data[l][2], data[l][3])
            M0[l] = metrics(n, sb, sbins, se); M[l] = metrics(n, sb, sbins, se, W)
        print(f'\n**Bootstrap over {C} {"benchmark" if ckey=="bench" else "subject-group"} clusters ({a.B} resamples, 95% percentile CI)**\n')
        print('| model | B0 | B1 | B3 | B7 | B15 | B31 | ALC [95% CI] | ECE [95% CI] | ECE@B0 | ECE@B31 | ECE (pooled responses) |'); print('|---|' + '---|' * 11)
        for l in labs:
            m0, m = M0[l], M[l]
            c = ci(m['ALC']); e = ci(m['ECE'])
            print(f"| {l} | " + ' | '.join(f'{x:.4f}' for x in m0['B'][0]) + f" | **{m0['ALC'][0]:.4f}** [{c[0]:.4f}, {c[1]:.4f}] | {m0['ECE'][0]:.4f} [{e[0]:.4f}, {e[1]:.4f}] | {m0['ECE0'][0]:.4f} | {m0['ECE31'][0]:.4f} | {m0['ECEpool'][0]:.4f} |")
        print('\nBrier per budget with 95% CI (pair-mean; the local analogue of the Codabench `brier` field, whose exact definition is UNKNOWN):\n')
        print('| model | ' + ' | '.join(f'B{n} [CI]' for n in H3.BUDGETS) + ' | ECE pooled responses [CI] |'); print('|---|' + '---|' * 7)
        for l in labs:
            cells = []
            for i in range(6):
                c = ci(M[l]['B'][:, i]); cells.append(f"{M0[l]['B'][0, i]:.4f} [{c[0]:.4f}, {c[1]:.4f}]")
            c = ci(M[l]['ECEpool']); print(f"| {l} | " + ' | '.join(cells) + f" | {M0[l]['ECEpool'][0]:.4f} [{c[0]:.4f}, {c[1]:.4f}] |")
        print('\n| paired delta (A - B), lower is better | dALC [95% CI] | P(dALC<0) | dECE [95% CI] | dECE pooled responses [95% CI] | dBrier B0 | dBrier B31 |'); print('|---|---|---|---|---|---|---|')
        pairs = [(l, a.ref) for l in labs if l != a.ref and a.ref in labs] + [(l, a.kit) for l in labs if l not in (a.kit,) and a.kit in labs and l != a.ref] + ([(a.ref, a.kit)] if a.ref in labs and a.kit in labs else [])
        for x, y in pairs:
            d = M[x]['ALC'] - M[y]['ALC']; de = M[x]['ECE'] - M[y]['ECE']; c = ci(d); ce = ci(de); cp = ci(M[x]['ECEpool'] - M[y]['ECEpool'])
            print(f"| {x} - {y} | {M0[x]['ALC'][0]-M0[y]['ALC'][0]:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {(d<0).mean():.3f} | {M0[x]['ECE'][0]-M0[y]['ECE'][0]:+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] | {M0[x]['ECEpool'][0]-M0[y]['ECEpool'][0]:+.4f} [{cp[0]:+.4f}, {cp[1]:+.4f}] | {M0[x]['B'][0,0]-M0[y]['B'][0,0]:+.4f} | {M0[x]['B'][0,5]-M0[y]['B'][0,5]:+.4f} |")
        print('\nPaired delta Brier per budget (A - B) [95% CI]:\n'); print('| A - B | ' + ' | '.join(f'B{n}' for n in H3.BUDGETS) + ' |'); print('|---|' + '---|' * 6)
        for x, y in pairs:
            cells = []
            for i in range(6):
                d = M[x]['B'][:, i] - M[y]['B'][:, i]; c = ci(d); cells.append(f"{M0[x]['B'][0, i]-M0[y]['B'][0, i]:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]")
            print(f'| {x} - {y} | ' + ' | '.join(cells) + ' |')
