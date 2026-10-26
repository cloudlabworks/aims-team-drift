"""Pick the v3 hyper-parameters for each OUTER fold f from inner (out-of-fold) runs only: mean over S in {10,3,10i} of the pair-mean ALC
of fold-j episodes (j != f) scored with the inner(f,j) prior.  Fold f's own benchmarks/subjects never enter its own selection.

  select.py --prefix t_ --out work/chosen_v3.json
"""
import argparse, glob, json, os, pickle, re, sys
import numpy as np
sys.path.insert(0, 'dev3'); import harness3 as H3

ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='t_'); ap.add_argument('--out', default='work/chosen_v3.json')
ap.add_argument('--S', default='10,3,10i'); a = ap.parse_args()
Ss = a.S.split(',')
tags = sorted({re.match(r'(.*)_S(10i|10|3)_oof_', os.path.basename(p)).group(1) for p in glob.glob(f'work/res3/{a.prefix}*_S{Ss[-1]}_oof_f4_j3.pkl')})  # last-written file => the config is complete
tab = {}
for t in tags:
    row = []
    for f in range(H3.K):
        v = []
        for S in Ss:
            B = np.concatenate([r['br'] for j in range(H3.K) if j != f for r in pickle.load(open(f'work/res3/{t}_S{S}_oof_f{f}_j{j}.pkl', 'rb'))])
            v.append(H3.alc(B).mean())
        row.append(np.mean(v))
    tab[t] = row
print('| config | ' + ' | '.join(f'f{f}' for f in range(H3.K)) + ' | mean |'); print('|---|' + '---|' * 6)
for t in sorted(tab, key=lambda t: np.mean(tab[t])):
    print(f'| {t} | ' + ' | '.join(f'{x:.5f}' for x in tab[t]) + f' | {np.mean(tab[t]):.5f} |')
chosen = {str(f): min(tab, key=lambda t: tab[t][f]) for f in range(H3.K)}
print('chosen per outer fold:', chosen)
json.dump(dict(chosen=chosen, table=tab), open(a.out, 'w'), indent=1)
