"""Grouped-by-benchmark K-fold evaluation of the full submission model (prior + test-time update)."""
import sys, os, json, time, pickle, importlib.util, argparse
import numpy as np, array
sys.path.insert(0,'dev'); sys.path.insert(0,'submission')
import harness as H, folds as FO, train_prior as T
import multiprocessing as mp
ap=argparse.ArgumentParser()
ap.add_argument('--S',type=str,default='10'); ap.add_argument('--lam_i',type=float,default=1e-2); ap.add_argument('--lam_s',type=float,default=1e-2)
ap.add_argument('--alpha',type=float,default=0.5); ap.add_argument('--sd',type=str,default='{}'); ap.add_argument('--temp',type=float,default=1.0)
ap.add_argument('--drop',type=float,default=0.0); ap.add_argument('--tag',type=str,default='run'); ap.add_argument('--model',type=str,default='submission/model.py')
ap.add_argument('--bint',type=int,default=0); ap.add_argument('--opts',type=str,default='{}'); ap.add_argument('--folds',type=str,default='0,1,2,3,4')
a=ap.parse_args()
sp=importlib.util.spec_from_file_location('sm',a.model); sm=importlib.util.module_from_spec(sp); sp.loader.exec_module(sm)
K=5; fl=FO.folds(K)
def get_art(fi,byb):
    p=f'work/artifacts/fold{fi}_{a.lam_i}_{a.lam_s}_{a.alpha}_{a.drop}{'_bint' if a.bint else ''}.npz'
    if os.path.exists(p):
        z=np.load(p); return dict(b=float(z['b']),ws=z['ws'],wi=z['wi'],idf=z['idf'])
    tr=[b for j in range(K) if j!=fi for b in fl[j]]
    art=T.fit_prior(tr,a.lam_i,a.lam_s,a.alpha,feats=[byb[b] for b in tr],drop_subj=a.drop,bench_int=bool(a.bint))
    os.makedirs('work/artifacts',exist_ok=True); np.savez(p,**art); return art
PRED=None
def work(ep):
    pb=lambda targets,L:[PRED.predict(t,L) for t in targets]
    return ep['bench'],H.run_episode(ep,pb)
if __name__=='__main__':
    eps=pickle.load(open(f'work/episodes/S{a.S}.pkl','rb'))
    allb=sum(fl,[]); byb=None
    res=[]; per_bench={}
    for fi in [int(x) for x in a.folds.split(',')]:
        if byb is None and not all(os.path.exists(f'work/artifacts/fold{j}_{a.lam_i}_{a.lam_s}_{a.alpha}_{a.drop}{'_bint' if a.bint else ''}.npz') for j in range(K)):
            byb={f['bench']:f for f in T.load_feats(allb)}
        art=get_art(fi,byb)
        meta=dict(b=art['b']); arr=dict(ws=array.array('f',art['ws'].tolist()),wi=array.array('f',art['wi'].tolist()),idf=array.array('f',art['idf'].tolist()))
        kw=dict(opts=json.loads(a.opts)) if 'opts' in sm.Predictor.__init__.__code__.co_varnames else {}
        PRED=sm.Predictor(meta,arr,prior_sd=json.loads(a.sd),temp=a.temp,**kw)
        ev=[e for e in eps if e['bench'] in set(fl[fi])]
        t0=time.time()
        with mp.get_context('fork').Pool(12) as p: out=p.map(work,ev,chunksize=2)
        B=np.concatenate([r for _,r in out]); res.append(B)
        for b,r in out: per_bench[b]=r
        print('fold',fi,'pairs',len(B),'budget Brier',B.mean(0).round(4),'ALC',H.alc(B).mean().round(4),f'{time.time()-t0:.0f}s',flush=True)
    folds_alc=[H.alc(B).mean() for B in res]; allB=np.concatenate(res)
    print(f'RESULT {a.tag} S={a.S} ALC {np.mean(folds_alc):.4f} +- {np.std(folds_alc):.4f} | per-budget',allB.mean(0).round(4),'| fold-std per budget',np.std([B.mean(0) for B in res],0).round(4))
    pickle.dump(dict(per_bench=per_bench,res=res,args=vars(a)),open(f'work/results/{a.tag}_S{a.S}.pkl','wb'))
