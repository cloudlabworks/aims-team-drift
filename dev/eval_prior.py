import sys, time, pickle, numpy as np
sys.path.insert(0,'dev'); sys.path.insert(0,'submission')
import harness as H, featurize as F, folds as FO, train_prior as T
def prior_logit(art, subject, item):
    tf=F.item_tf(item); idx=list(tf); v=np.array([tf[j] for j in idx])*art['idf'][idx]
    v=v/ (np.linalg.norm(v)+1e-9); zi=float(v@art['wi'][idx])
    zs=float(art['ws'][F.subject_idx(subject)].sum())
    return art['b']+zs+zi
def score(art, eps):
    out=[]
    for ep in eps:
        def pb(targets,L): return [1/(1+np.exp(-prior_logit(art,t[0],t[1]))) for t in targets]
        out.append(H.run_episode(ep,pb)[:,0])   # budget 0 only
    return out
if __name__=='__main__':
    S=int(sys.argv[1]); lam_i=float(sys.argv[2]); lam_s=float(sys.argv[3]); alpha=float(sys.argv[4]); K=5
    eps=pickle.load(open(f'work/episodes/S{S}.pkl','rb'))
    fl=FO.folds(K); res=[]
    allf=T.load_feats(sum(fl,[]))
    byb={f['bench']:f for f in allf}
    for fi in range(K):
        tr=[b for j in range(K) if j!=fi for b in fl[j]]
        t0=time.time(); art=T.fit_prior(tr,lam_i,lam_s,alpha,feats=[byb[b] for b in tr],verbose=True)
        ev=[e for e in eps if e['bench'] in set(fl[fi])]
        briers=np.concatenate(score(art,ev)) if ev else np.array([])
        # note run_episode computes all budgets; here preds ignore labels -> budget0 col only
        res.append(briers.mean()); print('fold',fi,len(briers),'brier0',round(briers.mean(),4),'t',round(time.time()-t0),flush=True)
    print('PRIOR S',S,lam_i,lam_s,alpha,'mean',np.mean(res).round(4),'std',np.std(res).round(4))
