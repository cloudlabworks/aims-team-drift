import os, json, numpy as np
# measurement-db README: release == 'withheld' (source not reproducible) -> excluded from training and evaluation
WITHHELD=set(k for k,v in json.load(open('work/release.json')).items() if v!='public')
def folds(K=5, seed=0):
    bs=sorted(f[:-4] for f in os.listdir('work/feats'))
    rng=np.random.default_rng(seed); perm=rng.permutation(len(bs))
    return [[bs[j] for j in perm[f::K] if bs[j] not in WITHHELD] for f in range(K)]
