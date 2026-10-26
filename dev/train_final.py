"""Fit the shipped prior on ALL public benchmarks and write submission/artifacts/."""
import sys, os, json, numpy as np
sys.path.insert(0,'dev'); sys.path.insert(0,'submission')
import train_prior as T
lam_i,lam_s,alpha=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
import folds as FO
bs=sorted(f[:-4] for f in os.listdir('work/feats') if f[:-4] not in FO.WITHHELD)
art=T.fit_prior(bs,lam_i,lam_s,alpha,verbose=True)
out='submission/artifacts'; os.makedirs(out,exist_ok=True)
json.dump(dict(b=art['b'],lam_i=lam_i,lam_s=lam_s,alpha=alpha,n_benchmarks=len(bs)),open(f'{out}/meta.json','w'))
for k in ('ws','wi','idf'): art[k].astype('<f4').tofile(f'{out}/{k}.f32')
print('written',{k:os.path.getsize(f'{out}/{k}.f32') for k in ('ws','wi','idf')})
