"""Evaluate a predictor on the unseen-subject diagonal split. Writes pair-level stats (+ raw per-target predictions) per fold.

  eval3.py --model v3 --tag NAME --S 10 [--mode outer|oof] [--opts JSON] [--sd JSON] [--calib PATH|none] [--folds 0,1,2,3,4]
models: const05 | empirical_mean (the kit's, unchanged) | v2 (submission/model.py, frozen shipped settings) | v3 (submission3/model.py)
mode outer: fold-f episodes scored with outer(f) prior.   mode oof: for every j != f, fold-j episodes scored with inner(f,j) prior.
Output: work/res3/{tag}_S{S}_{mode}_f{f}[_j{j}].pkl = list of per-episode dicts (bench, groups, br, bins, det)
"""
import argparse, array, importlib.util, json, os, pickle, sys, time
import numpy as np
import multiprocessing as mp
sys.path.insert(0, 'dev'); sys.path.insert(0, 'dev3'); sys.path.insert(0, 'submission3')
import harness3 as H3, priors as PR

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--S', default='10')
ap.add_argument('--mode', default='outer'); ap.add_argument('--opts', default='{}'); ap.add_argument('--sd', default='{}')
ap.add_argument('--calib', default='none'); ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--temp', type=float, default=1.0)
ap.add_argument('--procs', type=int, default=12); ap.add_argument('--clip', type=float, default=0.01); ap.add_argument('--oldprior', action='store_true')
A = ap.parse_args()
OUTD = 'work/res3'; os.makedirs(OUTD, exist_ok=True)
EMB = None
PRED = None


def load_module(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def make_predict_batch(art, calib):
    global EMB
    if A.model == 'const05':
        return lambda t, L: [0.5] * len(t)
    if A.model == 'empirical_mean':
        fn = load_module('paiec_baseline/empirical_mean/model.py', 'em').predict
        return lambda t, L: [fn(x, L) for x in t]
    meta = dict(b=float(art['b']))
    arr = dict(ws=array.array('f', art['ws'].tolist()), wi=array.array('f', art['wi'].tolist()), idf=array.array('f', art['idf'].tolist()))
    sd = json.loads(A.sd)
    if A.model == 'v2':
        sm = load_module('submission/model.py', 'sm2')
        P = sm.Predictor(meta, arr, prior_sd=sd, temp=A.temp, clip=(A.clip, 1 - A.clip))
    else:
        sm = load_module('submission3/model.py', 'sm3')
        arr['pca_mean'] = array.array('f', art['pca_mean'].tolist())
        arr['pca_basis'] = array.array('f', art['pca_basis'].reshape(-1).tolist())
        if EMB is None:
            EMB = pickle.load(open('work/emb_episode.pkl', 'rb'))
        emb = lambda it: EMB[(it['item_content'], it['item_features'])].tolist()
        P = sm.Predictor(meta, arr, prior_sd=sd, temp=A.temp, opts=json.loads(A.opts), calib=calib, embed=emb, clip=(A.clip, 1 - A.clip))
    return lambda t, L: [P.predict(x, L) for x in t]


def work(ep):
    det = H3.run_episode_detail(ep, PRED)
    br, bins = H3.summarize(ep, det)
    return dict(bench=ep['bench'], fold=ep['fold'], groups=[p['group'] for p in ep['pairs']], br=br, bins=bins, det=det)


def run(eps_fold, art, calib, outpath):
    global PRED
    PRED = make_predict_batch(art, calib)
    t0 = time.time()
    with mp.get_context('fork').Pool(A.procs) as p:
        res = p.map(work, eps_fold, chunksize=1)
    pickle.dump(res, open(outpath, 'wb'), protocol=4)
    B = np.concatenate([r['br'] for r in res])
    print(f"{os.path.basename(outpath)} pairs {len(B)} brier/budget {B.mean(0).round(4)} ALC {H3.alc(B).mean():.4f} {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    eps = pickle.load(open(f'work/episodes3/S{A.S}.pkl', 'rb'))
    byf = {f: [e for e in eps if e['fold'] == f] for f in range(H3.K)}
    calib = None if A.calib == 'none' else json.load(open(A.calib))
    base = f'{OUTD}/{A.tag}_S{A.S}'
    for f in [int(x) for x in A.folds.split(',')]:
        cal = calib[str(f)] if calib and str(f) in calib else calib
        if A.mode == 'outer':
            art = PR.get_prior(f) if A.model not in ('const05', 'empirical_mean') else None
            if A.oldprior:   # CONTROL: v2's original benchmark-grouped priors (trained on ALL subjects, incl. the held-out ones)
                z = np.load(f'work/artifacts/fold{f}_0.01_0.01_0.5_0.0.npz'); art = {k: z[k] for k in z.files}
            run(byf[f], art, cal, f'{base}_outer_f{f}.pkl')
        else:
            for j in range(H3.K):
                if j == f:
                    continue
                art = PR.get_prior(f, j)
                run(byf[j], art, None, f'{base}_oof_f{f}_j{j}.pkl')
