"""Per-benchmark hashed features for training (cached in work/feats)."""
import os, sys, pickle, numpy as np, scipy.sparse as sp
sys.path.insert(0,'submission'); import featurize as F
from multiprocessing import Pool
IN='work/cache'; OUT='work/feats'; MAXITEMS=10000
def do(b):
    d=pickle.load(open(f'{IN}/{b}.pkl','rb'))
    rng=np.random.default_rng(abs(hash(b))%2**32 if False else int.from_bytes(b.encode()[:4].ljust(4,b'0'),'big'))
    nI=len(d['items']); keep=np.sort(rng.permutation(nI)[:MAXITEMS])
    # carry interactors of first row per item if available
    inter={}
    if d['inter'] is not None:
        for j,i in enumerate(d['ii']):
            if i not in inter and d['inter'][j]: inter[int(i)]=d['inter'][j]
    ind=[];dat=[];ptr=[0]
    for i in keep:
        it=dict(d['items'][i]); it['interactors']=inter.get(int(i),'')
        tf=F.item_tf(it); ind.extend(tf.keys()); dat.extend(tf.values()); ptr.append(len(ind))
    X=sp.csr_matrix((np.array(dat,np.float32),np.array(ind,np.int32),np.array(ptr)),shape=(len(keep),F.D_ITEM))
    Sx=[F.subject_idx(s) for s in d['subjects']]
    sptr=np.cumsum([0]+[len(x) for x in Sx]); sind=np.concatenate([np.array(x,np.int32) for x in Sx])
    S=sp.csr_matrix((np.ones(len(sind),np.float32),sind,sptr),shape=(len(Sx),F.D_SUBJ))
    # rows restricted to kept items
    pos=np.full(nI,-1,np.int64); pos[keep]=np.arange(len(keep))
    m=pos[d['ii']]>=0
    rows=dict(si=d['si'][m],ii=pos[d['ii'][m]].astype(np.int32),n=d['n'][m],k=d['k'][m])
    pickle.dump(dict(bench=b,X=X,S=S,**rows),open(f'{OUT}/{b}.pkl','wb'),protocol=4)
    return b,len(keep),len(rows['si'])
if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    bs=sorted(f[:-4] for f in os.listdir(IN))
    with Pool(12) as p:
        for r in p.imap_unordered(do,bs): print(*r,flush=True)
