"""Local mimic of the PAIEC streaming ALC protocol (benchmark-level cold start).

Episodes: for one held-out benchmark, S sampled subjects (pairs with >=80 distinct items), shared item pool
capped so S*pool <= 1000 unique subject-item pairs (acq+eval, as in the formative sample cap), 50/50 per-pair
item split, random acquisition order with the kit's random-policy rule, nested label sets at budgets 0..31.
"""
import os, sys, json, pickle, hashlib, math
import numpy as np
BUDGETS=[0,1,3,7,15,31]; W=[0.1,0.2,0.2,0.2,0.2,0.1]
CACHE='work/cache'; EPI='work/episodes'
def load(b): return pickle.load(open(f'{CACHE}/{b}.pkl','rb'))
def subj_dict(d,i): return dict(d['subjects'][i])
def item_dict(d,i,bid,inter=''):
    x=d['items'][i]; return {'item_content':x['item_content'],'item_features':x['item_features'],'interactors':inter,'benchmark_id':bid}

def make_episode(b, S, seed, cap=1000, min_items=80, indep=False):
    d=load(b); rng=np.random.default_rng(seed)
    nS=len(d['subjects'])
    # per-subject item index -> (n,k,inter)
    order=np.argsort(d['si'],kind='stable'); si=d['si'][order]
    starts=np.searchsorted(si,np.arange(nS+1))
    elig=[s for s in range(nS) if starts[s+1]-starts[s]>=min_items]
    if not elig: return None
    chosen=list(rng.permutation(elig)[:S])
    pool_n=max(min_items,cap//len(chosen))
    first=chosen[0]; fi=d['ii'][order][starts[first]:starts[first+1]]
    pool=set(rng.permutation(fi)[:pool_n].tolist())
    bid='benchmark_'+hashlib.sha1(f'{b}|{seed}'.encode()).hexdigest()[:8]
    pairs=[]
    for s in chosen:
        sl=slice(starts[s],starts[s+1]); ii=d['ii'][order][sl]; n=d['n'][order][sl]; k=d['k'][order][sl]
        inter=[d['inter'][j] for j in order[sl]] if d['inter'] is not None else None
        if indep:
            take=rng.permutation(len(ii))[:pool_n]; sel=[int(j) for j in take]
        else:
            sel=[j for j in range(len(ii)) if ii[j] in pool]
        if len(sel)<min_items: continue
        sel=np.array(sel); perm=rng.permutation(len(sel)); half=len(sel)//2
        acq=sel[perm[:half]]; ev=sel[perm[half:]]
        # acquisition under kit random policy, candidate order = acq order, up to 31 labels
        got=[]; remaining=31
        for pos,j in enumerate(acq):
            items_remaining=len(acq)-pos
            if remaining<=0: break
            if rng.random()<min(1.0,remaining/items_remaining):
                got.append(j); remaining-=1
        def lab(j):  # reveal one recorded response
            return int(rng.random()<k[j]/n[j])
        labels=[(int(ii[j]), (inter[j] if inter else ''), lab(j)) for j in got]
        evl=[(int(ii[j]),(inter[j] if inter else ''),int(n[j]),int(k[j])) for j in ev]
        pairs.append(dict(subject=subj_dict(d,s),sid=int(s),labels=labels,eval=evl))
    if not pairs: return None
    used={x[0] for p in pairs for x in p['labels']+p['eval']}
    items={i:item_dict(d,i,bid,'') for i in used}
    return dict(bench=b,S=S,seed=seed,bid=bid,pairs=pairs,items={i:items[i] for i in used})

def build_all(S,draws=1,indep=False):
    os.makedirs(EPI,exist_ok=True)
    cat=sorted(f[:-4] for f in os.listdir(CACHE))
    out=[]
    for b in cat:
        for dr in range(draws):
            seed=int(hashlib.sha1(f'{b}|{S}|{dr}'.encode()).hexdigest()[:8],16)
            e=make_episode(b,S,seed,indep=indep)
            if e: out.append(e)
    pickle.dump(out,open(f"{EPI}/S{S}{'i' if indep else ''}.pkl",'wb'),protocol=4)
    return out

def inp(ep,pair,item_i,inter):
    it=dict(ep['items'][item_i]); it['interactors']=inter
    return [dict(pair['subject']),it]

def labeled_at(ep,n):
    L=[]
    for p in ep['pairs']:
        for (ii,inter,y) in p['labels'][:n]:
            L.append([inp(ep,p,ii,inter),y])
    return L

def run_episode(ep, predict_batch):
    """predict_batch(list_of_inputs, labeled) -> list of floats. Returns per-pair per-budget Brier array [P,6]."""
    P=len(ep['pairs']); out=np.zeros((P,len(BUDGETS)))
    for bi,n in enumerate(BUDGETS):
        L=labeled_at(ep,n)
        targets=[]; idx=[]
        for pi,p in enumerate(ep['pairs']):
            for (ii,inter,nn,kk) in p['eval']:
                targets.append(inp(ep,p,ii,inter)); idx.append(pi)
        preds=predict_batch(targets,L)
        tot=np.zeros(P); cnt=np.zeros(P)
        t=0
        for pi,p in enumerate(ep['pairs']):
            for (ii,inter,nn,kk) in p['eval']:
                q=float(preds[t]); t+=1
                assert 0<=q<=1 and math.isfinite(q)
                tot[pi]+=kk*(1-q)**2+(nn-kk)*q**2; cnt[pi]+=nn
        out[:,bi]=tot/cnt
    return out
def alc(b): return (np.asarray(b)*np.array(W)).sum(-1)
