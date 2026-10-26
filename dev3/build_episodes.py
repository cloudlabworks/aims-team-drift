import sys, time, collections
sys.path.insert(0,'dev3'); import harness3 as H3
for S,indep in [(10,False),(3,False),(10,True)]:
    t=time.time(); eps=H3.build_all(S,draws=5,indep=indep)
    tag=f"S{S}{'i' if indep else ''}"
    byf=collections.defaultdict(lambda:[0,0]); 
    for e in eps: byf[e['fold']][0]+=1; byf[e['fold']][1]+=len(e['pairs'])
    print(tag,'episodes',len(eps),'pairs',sum(len(e['pairs']) for e in eps),'by fold (eps,pairs)',dict(sorted(byf.items())),f'{time.time()-t:.0f}s',flush=True)
