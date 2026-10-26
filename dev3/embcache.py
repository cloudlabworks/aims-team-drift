"""Embedding caches for the local harness (the shipped stdlib encoder's tokenizer + its float16 table, mean-pooled with numpy).

  python dev3/embcache.py train      -> work/emb_train.npz   (<=300 items per public benchmark, for the per-fit PCA bases)
  python dev3/embcache.py episodes   -> work/emb_episode.pkl {(item_content, item_features): float32[256]} for every episode item
Equality with the stdlib Encoder.encode is asserted on a sample.
"""
import os, pickle, sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, 'submission3'); sys.path.insert(0, 'dev')
import embed as E

_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        e = E.Encoder('submission3/artifacts')
        e.table = np.frombuffer(e.buf, dtype='<f2').reshape(-1, E.DIM).astype(np.float32)
        _ENC = e
    return _ENC


def encode_fast(text):
    e = enc(); ids = e.token_ids(text)
    if not ids:
        return np.zeros(E.DIM, np.float32)
    v = e.table[ids].mean(0); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-12 else np.zeros(E.DIM, np.float32)


def _train_one(b, per=300, seed=0):
    import harness as H
    d = H.load(b); rng = np.random.default_rng(int.from_bytes(b.encode()[:4].ljust(4, b'0'), 'big') + seed)
    idx = rng.permutation(len(d['items']))[:per]
    return b, np.stack([encode_fast(E.item_text(d['items'][i])) for i in idx])


def _enc_keys(keys):
    return [encode_fast(((c or '')[:E.MAX_CHARS] + ' ' + (f or '')[:300]).strip()) for c, f in keys]


def check_equal(keys):
    e = enc()
    for c, f in keys[:200]:
        t = ((c or '')[:E.MAX_CHARS] + ' ' + (f or '')[:300]).strip()
        a, b = np.array(e.encode(t)), encode_fast(t)
        assert np.allclose(a, b, atol=1e-5), t[:60]


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'train':
        rel = json.load(open('work/release.json'))
        bs = sorted(f[:-4] for f in os.listdir('work/cache') if rel.get(f[:-4]) == 'public')
        with Pool(6) as p:
            res = dict(p.imap_unordered(_train_one, bs))
        np.savez('work/emb_train.npz', **res); print('benchmarks', len(res), 'items', sum(len(v) for v in res.values()))
    else:
        keys = set()
        for tag in ('S10', 'S3', 'S10i'):
            for ep in pickle.load(open(f'work/episodes3/{tag}.pkl', 'rb')):
                for it in ep['items'].values():
                    keys.add((it['item_content'], it['item_features']))
        keys = sorted(keys); print('unique episode items', len(keys), flush=True)
        check_equal(keys)
        chunks = [keys[i::60] for i in range(60)]
        with Pool(8) as p:
            outs = p.map(_enc_keys, chunks)
        cache = {}
        for ch, o in zip(chunks, outs):
            cache.update(zip(ch, o))
        pickle.dump(cache, open('work/emb_episode.pkl', 'wb'), protocol=4); print('cached', len(cache))
