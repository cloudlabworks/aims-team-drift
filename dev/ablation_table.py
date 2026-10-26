import sys, os, pickle, numpy as np
sys.path.insert(0,'dev'); import harness as H, folds as FO
fl=FO.folds(5); bf={b:i for i,f in enumerate(fl) for b in f}
def stats(nm,S='10'):
    d=pickle.load(open(f'work/results/{nm}_S{S}.pkl','rb')); pb=d['per_bench'] if 'per_bench' in d else d
    per=[[] for _ in range(5)]
    for b,r in pb.items():
        if b in bf: per[bf[b]].append(r)
    per=[np.concatenate(x) for x in per]
    return np.array([x.mean(0) for x in per]).mean(0), np.mean([H.alc(x).mean() for x in per])
ref,refa=stats('final')
rows=[('m1','test-time d-prior sd 1.0 (earlier default)'),('d0.5','d sd 0.5'),('d1.5','d sd 1.5 (shipped)'),('d2.5','d sd 2.5'),('beta1.0','beta sd 1.0'),('beta2.5','beta sd 2.5'),
 ('delta0.4','pair-offset sd 0.4'),('delta1.5','pair-offset sd 1.5'),('g0.15','slope sd 0.15'),('g0.7','slope sd 0.7'),('temp0.9','temperature 0.9'),('temp1.15','temperature 1.15'),
 ('ev1','per-benchmark evidence grid over (d, pair) sds'),('var_du0.5','probit variance shrinkage du=0.5'),('var_du0.5_half','... half strength'),
 ('k1','+ text-kernel residual smoothing c=3'),('kern2','+ kernel c=6'),
 ('ss0','NO subject prior (zs=0)'),('si0','NO item-text prior (zi=0)'),('ss1.5','subject prior x1.5'),
 ('lam_i3e-3','prior lam_i 3e-3'),('lam_i1e-1','prior lam_i 1e-1'),('lam_s1e-3','prior lam_s 1e-3'),('bint1','prior with benchmark intercepts'),('li1e-5b','weak reg lam_i 1e-5 + bench intercepts'),('li1e-6','weak reg lam_i 1e-6, lam_s 1e-4')]
print('| variant | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ΔALC vs shipped |'); print('|---|'+'---|'*8)
for tag,desc in rows:
    if not os.path.exists(f'work/results/{tag}_S10.pkl'): continue
    m,a=stats(tag); print(f'| {desc} | '+' | '.join(f'{x:.4f}' for x in m)+f' | {a:.4f} | {a-refa:+.4f} |')
