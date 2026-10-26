"""Compact per-benchmark cache: aggregated (subject,item) trial/success counts + input strings."""
import os, sys, json, pickle
import numpy as np, pandas as pd, pyarrow.parquet as pq
from multiprocessing import Pool
ROOT='data/measurement-db'; OUT='work/cache'
SK=['normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra']
def s(x, lim):
    if x is None or (isinstance(x,float) and np.isnan(x)): return ''
    return str(x)[:lim]
def do(b):
    try:
        p=f'{ROOT}/{b}/formatted_tables/'
        r=pq.read_table(p+'responses.parquet',columns=['subject_id','item_id','response','interactors']).to_pandas()
        r=r[r.response.notna()]
        if len(r)==0 or not set(np.unique(r.response.values)).issubset({0.0,1.0}): return b,'nonbinary-or-empty'
        r['response']=r.response.astype('int8')
        g=r.groupby(['subject_id','item_id'],sort=True)
        agg=g.agg(n=('response','size'),k=('response','sum'),inter=('interactors','first')).reset_index()
        subj=pd.read_parquet(p+'subjects.parquet')
        it=pq.read_table(p+'items.parquet',columns=['item_id','content','item_features']).to_pandas()
        su=sorted(agg.subject_id.unique()); iu=sorted(agg.item_id.unique())
        sm={x:i for i,x in enumerate(su)}; im={x:i for i,x in enumerate(iu)}
        subj=subj.set_index('subject_id').loc[su]; it=it.drop_duplicates('item_id').set_index('item_id').reindex(iu)
        subjects=[{k:s(v,400) for k,v in zip(SK,row)} for row in subj[SK].itertuples(index=False)]
        items=[{'item_content':s(a,3000),'item_features':s(c,600)} for a,c in zip(it.content.values,it.item_features.values)]
        d=dict(bench=b,subjects=subjects,items=items,
               si=agg.subject_id.map(sm).values.astype(np.int32),ii=agg.item_id.map(im).values.astype(np.int32),
               n=agg.n.values.astype(np.int32),k=agg.k.values.astype(np.int32),
               inter=[s(x,300) for x in agg.inter.values] if agg.inter.notna().any() else None)
        pickle.dump(d,open(f'{OUT}/{b}.pkl','wb'),protocol=4)
        return b,f'ok rows={len(agg)} S={len(su)} I={len(iu)}'
    except Exception as e:
        return b,'ERR '+repr(e)[:200]
if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    cat=json.load(open('work/catalog.json'))
    bs=[k for k,v in cat.items() if v['binary']]
    big=sorted(bs,key=lambda k:-cat[k]['n'])
    with Pool(8) as pool:
        for b,m in pool.imap_unordered(do,big):
            print(b,m,flush=True)
