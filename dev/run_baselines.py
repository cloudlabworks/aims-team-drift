import sys, os, json, pickle, importlib.util, numpy as np
sys.path.insert(0,'dev'); import harness as H
from multiprocessing import Pool
def load_predict(path,name):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m.predict
def adapter(fn):
    def pb(targets,L): return [fn(t,L) for t in targets]
    return pb
MODELS={'const05':lambda: (lambda t,L:0.5),
        'empirical_mean':lambda: load_predict('paiec_baseline/empirical_mean/model.py','em'),
        'stdlib_baseline':lambda: load_predict('baselines/stdlib_model.py','sb')}
def work(a):
    name,ep=a; return ep['bench'],H.run_episode(ep,adapter(MODELS[name]()))
if __name__=='__main__':
    name,S=sys.argv[1],sys.argv[2]
    eps=pickle.load(open(f'work/episodes/S{S}.pkl','rb'))
    with Pool(12) as p: res=p.map(work,[(name,e) for e in eps],chunksize=1)
    os.makedirs('work/results',exist_ok=True)
    pickle.dump({b:r for b,r in res},open(f'work/results/{name}_S{S}.pkl','wb'))
    B=np.concatenate([r for _,r in res]); print(name,S,'pairs',len(B),'brier by budget',B.mean(0).round(4),'ALC',H.alc(B).mean().round(4))
