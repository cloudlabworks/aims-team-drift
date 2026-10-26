"""Write the bundled static-encoder artifacts (vocab + float16 table) from the local potion-base-8M copy."""
import json, struct, numpy as np
D='work/models/potion-base-8M'; O='submission3/artifacts'
t=json.load(open(D+'/tokenizer.json'))['model']['vocab']
vocab=sorted(t,key=t.get); assert [t[w] for w in vocab]==list(range(len(vocab)))
open(O+'/embed_vocab.txt','w',encoding='utf-8').write('\n'.join(vocab)+'\n')
b=open(D+'/model.safetensors','rb'); n=struct.unpack('<Q',b.read(8))[0]; h=json.loads(b.read(n)); off=h['embeddings']['data_offsets']
E=np.frombuffer(b.read()[off[0]:off[1]],dtype='<f4').reshape(h['embeddings']['shape'])
assert E.shape[0]==len(vocab)
E.astype('<f2').tofile(O+'/embed_table.f16'); print('vocab',len(vocab),'table',E.shape)
