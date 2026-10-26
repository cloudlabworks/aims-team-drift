import sys, pickle, numpy as np, array, importlib.util
sys.path.insert(0,'dev'); sys.path.insert(0,'submission')
import folds as FO
from scipy.stats import spearmanr
sp=importlib.util.spec_from_file_location('sm','submission/model.py'); sm=importlib.util.module_from_spec(sp); sp.loader.exec_module(sm)
eps=pickle.load(open('work/episodes/S10.pkl','rb')); fl=FO.folds(5)
tag=sys.argv[1] if len(sys.argv)>1 and sys.argv[1]!='-' else ''; lam=sys.argv[2] if len(sys.argv)>2 else '0.01_0.01'
rs=[];ws=[]
for fi in range(5):
    z=np.load(f'work/artifacts/fold{fi}_{lam}_0.5_0.0{tag}.npz')
    P=sm.Predictor(dict(b=float(z['b'])),dict(ws=array.array('f',z['ws'].tolist()),wi=array.array('f',z['wi'].tolist()),idf=array.array('f',z['idf'].tolist())))
    for e in eps:
        if e['bench'] not in fl[fi]: continue
        acc={}
        for p in e['pairs']:
            for (ii,inter,n,k) in p['eval']:
                a=acc.setdefault(ii,[0,0]); a[0]+=k; a[1]+=n
        ids=[i for i in acc if acc[i][1]>0]
        if len(ids)<30: continue
        y=np.array([acc[i][0]/acc[i][1] for i in ids]); x=np.array([P.zi(e['items'][i]) for i in ids])
        if y.std()>0 and x.std()>0: rs.append(spearmanr(x,y)[0]); 
print(tag,'benchmarks',len(rs),'mean spearman(zi, item success)',np.nanmean(rs).round(3),'median',np.nanmedian(rs).round(3),'frac>0',np.nanmean(np.array(rs)>0).round(2))
