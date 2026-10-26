import sys, pickle, numpy as np
sys.path.insert(0,'dev'); import harness as H, folds as FO
def table(names,S,K=5):
    fl=FO.folds(K); bf={b:i for i,f in enumerate(fl) for b in f}
    rows=[]
    for nm in names:
        d=pickle.load(open(f'work/results/{nm}_S{S}.pkl','rb')); pb=d['per_bench'] if 'per_bench' in d else d
        per=[[] for _ in range(K)]
        for b,r in pb.items():
            if b in bf: per[bf[b]].append(r)
        per=[np.concatenate(x) for x in per if len(x)]
        fm=np.array([x.mean(0) for x in per]); alc=np.array([H.alc(x).mean() for x in per])
        allB=np.concatenate(per)
        rows.append((nm,fm.mean(0),fm.std(0),alc.mean(),alc.std(),len(allB)))
    print(f'### S={S} (subjects per benchmark); cells = mean ± std over {K} benchmark-folds')
    print('| model | B0 | B1 | B3 | B7 | B15 | B31 | **ALC** | pairs |'); print('|---|'+'---|'*8)
    for nm,m,s,a,sa,n in rows:
        print(f'| {nm} | '+' | '.join(f'{x:.3f}±{y:.3f}' for x,y in zip(m,s))+f' | **{a:.4f}±{sa:.4f}** | {n} |')
if __name__=='__main__':
    table(sys.argv[2:],sys.argv[1])
