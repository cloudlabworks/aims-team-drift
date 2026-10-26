"""Validate the shipped stdlib encoder against the model2vec reference (run with a python that has model2vec)."""
import sys, pickle, os, random, numpy as np
sys.path.insert(0,'submission3'); import embed as E
from model2vec import StaticModel
ref=StaticModel.from_pretrained('work/models/potion-base-8M')
enc=E.Encoder('submission3/artifacts')
texts=['Hello, World!','','   ','Café déjà vu — naïve résumé','x'*150,'中文测试 and English','What is 2+2? (A) 3 (B) 4 [SEP] [CLS] ##ing','print("hi")\nfor i in range(10):\n\tpass','​zero width','<|endoftext|> 3.14159e-10 $5,000.00','ß ǆ İstanbul ﬁne','a'*3000]
random.seed(0)
for b in ['abstract_reason','adaptivestep','aegis','agentdojo','ceval','chi_bench','beavertails','bird_sql']:
    d=pickle.load(open(f'work/cache/{b}.pkl','rb'))
    for it in random.sample(d['items'],min(60,len(d['items']))): texts.append(E.item_text(it))
cs=[];bad=0
for t in texts:
    a=np.array(enc.encode(t)); r=ref.encode([t[:E.MAX_CHARS]],max_length=E.MAX_TOKENS)[0]
    r=r/ (np.linalg.norm(r)+1e-12) if np.linalg.norm(r)>0 else r
    if np.linalg.norm(a)==0 and np.linalg.norm(r)==0: cs.append(1.0); continue
    c=float(a@r/(np.linalg.norm(a)*np.linalg.norm(r)+1e-12)); cs.append(c)
    if c<0.999: bad+=1; print('LOW',round(c,5),repr(t[:80]))
print('texts',len(texts),'min cos',min(cs),'mean cos',np.mean(cs),'n<0.999',bad)
