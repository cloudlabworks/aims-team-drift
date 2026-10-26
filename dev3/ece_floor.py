"""ECE noise floor: ECE of the SAME predictions if they were perfectly calibrated (responses resampled as Binomial(n, p)),
computed exactly as the pair-mean ECE in analyze.py. Shows how much of the measured ECE is finite-sample noise / unavoidable."""
import pickle, sys
import numpy as np
sys.path.insert(0, 'dev3'); import harness3 as H3
S, tag = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0); acc, accb = [], []
for f in range(H3.K):
    for r in pickle.load(open(f'work/res3/{tag}_S{S}_outer_f{f}.pkl', 'rb')):
        d = r['det']; P = r['br'].shape[0]; key = d['pair'] * 6 + d['bud']
        k = rng.binomial(d['n'], d['p'].astype(float)).astype(float); n = d['n'].astype(float); p = d['p'].astype(float)
        b = np.minimum((p * 10).astype(int), 9); bins = np.zeros((P * 6, 10, 3))
        for c, w in enumerate((n, n * p, k)): np.add.at(bins[:, :, c], (key, b), w)
        e = H3.ece(bins.reshape(P, 6, 10, 3)); acc.append(e)
E = np.concatenate(acc); print(f'S={S} {tag}: floor ECE (pair-mean, all budgets) {E.mean():.4f} | B0 {E[:,0].mean():.4f} | B31 {E[:,5].mean():.4f}')
