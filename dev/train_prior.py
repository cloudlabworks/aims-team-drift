"""Offline prior: logit p = b + ws.phi(subject) + wi.psi(item), weighted logistic ridge fitted across benchmarks."""
import os, sys, pickle, numpy as np, scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b
sys.path.insert(0,'submission'); import featurize as F
FE='work/feats'
def load_feats(benches):
    return [pickle.load(open(f'{FE}/{b}.pkl','rb')) for b in benches]
def fit_prior(benches, lam_i=1.0, lam_s=1.0, alpha=0.5, maxiter=120, drop_subj=0.0, seed=0, feats=None, verbose=False, bench_int=False, hide_fn=None):
    fs=feats if feats is not None else load_feats(benches)
    rng=np.random.default_rng(seed)
    Xs=[];Ss=[];si=[];ii=[];n=[];k=[];wt=[];bi=[]; ioff=0; soff=0
    for f in fs:
        Xs.append(f['X']); Ss.append(f['S'])
        s_,i_,n_,k_=f['si'],f['ii'],f['n'],f['k']
        if drop_subj>0:   # hide a share of subject identities from training
            hide=rng.random(f['S'].shape[0])<drop_subj
            m=~hide[s_]; s_,i_,n_,k_=s_[m],i_[m],n_[m],k_[m]
        if hide_fn is not None:   # v3 unseen-subject split: hide_fn(bench) -> bool[n_subjects], True = drop that subject's rows
            m=~hide_fn(f['bench'])[s_]; s_,i_,n_,k_=s_[m],i_[m],n_[m],k_[m]
        si.append(s_+soff); ii.append(i_+ioff); n.append(n_); k.append(k_)
        tot=n_.sum(); wt.append(np.full(len(s_), (tot**(alpha-1.0)) if tot>0 else 0.0))
        bi.append(np.full(len(s_),len(bi),np.int32))
        ioff+=f['X'].shape[0]; soff+=f['S'].shape[0]
    X=sp.vstack(Xs).tocsr(); S=sp.vstack(Ss).tocsr()
    si=np.concatenate(si);ii=np.concatenate(ii);n=np.concatenate(n).astype(np.float64);k=np.concatenate(k).astype(np.float64);wt=np.concatenate(wt)
    df=np.bincount(X.indices,minlength=F.D_ITEM).astype(np.float64); N=X.shape[0]
    idf=(np.log((N+1.0)/(df+1.0))+1.0).astype(np.float32)
    X=X.multiply(idf[None,:]).tocsr()
    nrm=np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())+1e-9
    X=sp.diags(1.0/nrm).dot(X).tocsr().astype(np.float32)
    bi=np.concatenate(bi); nb=len(fs)
    w=wt*n; wk=wt*k
    Wn=w.sum()
    XT=X.T.tocsr(); ST=S.T.tocsr()
    nD,sD=F.D_ITEM,F.D_SUBJ
    def f_g(theta):
        b=theta[0]; ws=theta[1:1+sD]; wi=theta[1+sD:1+sD+nD]; bb=theta[1+sD+nD:]
        z=(S@ws)[si]+(X@wi)[ii]+b
        if bench_int: z=z+bb[bi]
        p=1/(1+np.exp(-np.clip(z,-30,30)))
        # loss per row: wt*(n*softplus(z) - k*z)
        sp_=np.logaddexp(0,z)
        loss=(wt*n*sp_).sum()-(wk*z).sum()
        r=(w*p-wk)           # d loss / dz
        gb=r.sum()
        gs=ST@np.bincount(si,weights=r,minlength=S.shape[0])
        gi=XT@np.bincount(ii,weights=r,minlength=N)
        loss=loss/Wn+0.5*lam_s*(ws@ws)/sD*0+0.5*lam_s*(ws@ws)+0.5*lam_i*(wi@wi)
        gbb=(np.bincount(bi,weights=r,minlength=nb)/Wn+1e-6*bb) if bench_int else np.zeros(nb)
        g=np.concatenate([[gb/Wn],gs/Wn+lam_s*ws,gi/Wn+lam_i*wi,gbb])
        return loss,g
    th0=np.zeros(1+sD+nD+nb)
    p0=k.sum()/n.sum(); th0[0]=np.log(p0/(1-p0))
    th,fv,info=fmin_l_bfgs_b(f_g,th0,maxiter=maxiter,m=20)
    if verbose: print('lbfgs',fv,info['nit'],info['warnflag'])
    bb=th[1+sD+nD:]
    b0=float(th[0])
    if bench_int:   # global intercept = benchmark-weighted mean of the benchmark intercepts
        tot=np.array([wt[bi==j].sum()*0+ (n[bi==j]*wt[bi==j]).sum() for j in range(nb)])
        b0=float(th[0]+(bb*tot).sum()/tot.sum())
    return dict(b=b0,ws=th[1:1+sD].astype(np.float32),wi=th[1+sD:1+sD+nD].astype(np.float32),idf=idf)
