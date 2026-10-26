"""Offline priors for the diagonal unseen-subject design (frozen v2 hyper-parameters: lam_i=1e-2, lam_s=1e-2, alpha=0.5).

outer(f)   : trained on benchmarks outside fold f, only rows of subject groups outside fold f.   -> scores fold-f episodes.
inner(f,j) : trained outside folds {f, j} for both benchmarks and subject groups.                 -> out-of-fold predictions on
             fold-j episodes, used ONLY for tuning/calibration of the outer-fold-f model (nothing of fold f is ever seen).
Each artifact also carries a PCA basis (whitened, 32 comps) of static item embeddings fitted on the same training benchmarks.
"""
import os, sys
import numpy as np
sys.path.insert(0, 'dev'); sys.path.insert(0, 'dev3')
import train_prior as T
import harness3 as H3

LAM = (1e-2, 1e-2, 0.5)
OUT = 'work/art3'
KMAX = 32
_EMB = None


def _emb():
    global _EMB
    if _EMB is None:
        _EMB = dict(np.load('work/emb_train.npz'))
    return _EMB


def pca_basis(benches, kmax=KMAX):
    X = np.concatenate([_emb()[b] for b in benches]).astype(np.float64)
    X = X[np.linalg.norm(X, axis=1) > 0]
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return mu.astype(np.float32), (Vt[:kmax] / np.sqrt(var[:kmax])[:, None]).astype(np.float32), var[:kmax]


def get_prior(f, j=None):
    path = f'{OUT}/outer{f}.npz' if j is None else f'{OUT}/inner{f}_{j}.npz'
    if os.path.exists(path):
        z = np.load(path); return {k: z[k] for k in z.files}
    os.makedirs(OUT, exist_ok=True)
    folds = {f} | (set() if j is None else {j})
    benches = sorted(b for k in range(H3.K) if k not in folds for b in H3.BFOLDS[k])
    drop = list(folds)
    art = T.fit_prior(benches, *LAM, hide_fn=lambda b: np.isin(H3.subject_fold(b), drop), verbose=True)
    mu, B, var = pca_basis(benches)
    art.update(pca_mean=mu, pca_basis=B, pca_var=var, n_benchmarks=np.array(len(benches)))
    np.savez(path, **art)
    return art


if __name__ == '__main__':
    import time
    jobs = [(f, None) for f in range(H3.K)] + [(f, j) for f in range(H3.K) for j in range(H3.K) if j != f]
    for f, j in jobs:
        t = time.time(); get_prior(f, j); print('prior', f, j, f'{time.time()-t:.0f}s', flush=True)
