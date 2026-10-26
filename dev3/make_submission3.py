"""Assemble submission3/artifacts for the shipped v3 model.

  make_submission3.py --k 8 --su 0.2 --d 1.5 --calib work/calib_final_v3.json

prior (ws, wi, idf, b): the SAME frozen v2 prior fitted on all 167 public binary benchmarks (submission/artifacts) - v3 changes only
the test-time layer, so no new prior is trained.  pca_mean / pca_basis: whitened PCA (32 comps) of static item embeddings of up to
300 items per public benchmark (work/emb_train.npz).  calib.json: the cross-fitted per-budget calibrator (dev3/calibrate.py --final).
"""
import argparse, json, shutil, sys
import numpy as np
sys.path.insert(0, 'dev3'); import priors as PR, harness3 as H3

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, required=True); ap.add_argument('--su', type=float, required=True); ap.add_argument('--d', type=float, default=1.5)
ap.add_argument('--calib'); a = ap.parse_args()
O = 'submission3/artifacts'
for n in ('ws', 'wi', 'idf'):
    shutil.copy(f'submission/artifacts/{n}.f32', f'{O}/{n}.f32')
meta = json.load(open('submission/artifacts/meta.json'))
benches = sorted(b for fl in H3.BFOLDS for b in fl)
assert len(benches) == meta['n_benchmarks'] == 167
mu, B, var = PR.pca_basis(benches)
mu.astype('<f4').tofile(f'{O}/pca_mean.f32'); B.astype('<f4').tofile(f'{O}/pca_basis.f32')
meta.update(opts=dict(k=a.k, su=a.su), prior_sd=dict(d=a.d), embedding='minishlab/potion-base-8M (bundled, stdlib encoder)', pca_var=[float(v) for v in var])
json.dump(meta, open(f'{O}/meta.json', 'w'), indent=1)
if a.calib:
    shutil.copy(a.calib, f'{O}/calib.json')
print('written', sorted(__import__('os').listdir(O)))
