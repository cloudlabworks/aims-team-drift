import sys, os, json, glob
import pyarrow.parquet as pq, pyarrow.compute as pc, numpy as np
root='data/measurement-db'
out={}
for d in sorted(os.listdir(root)):
    f=f'{root}/{d}/formatted_tables/responses.parquet'
    if not os.path.exists(f): continue
    try:
        t=pq.read_table(f,columns=['subject_id','item_id','response','trial','test_condition'])
        r=t.column('response').to_numpy(zero_copy_only=False)
        u=np.unique(r[~np.isnan(r)]) if len(r) else r
        nan=int(np.isnan(r).sum())
        sub=pc.count_distinct(t.column('subject_id')).as_py(); it=pc.count_distinct(t.column('item_id')).as_py()
        out[d]=dict(n=t.num_rows,subj=sub,items=it,nuniq=int(len(u)),vals=[float(x) for x in u[:6]],nan=nan,
                    binary=bool(len(u)>0 and set(u.tolist())<= {0.0,1.0}),
                    mean=float(np.nanmean(r)) if len(r) else None,
                    trials=int(pc.max(t.column('trial')).as_py() or 0),
                    ntc=pc.count_distinct(t.column('test_condition')).as_py())
    except Exception as e:
        out[d]=dict(error=str(e)[:200])
json.dump(out,open('work/catalog.json','w'),indent=1)
print(len(out))
