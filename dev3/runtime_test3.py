"""Fresh-process timing of the shipped v3 zip (extracted dir) + equivalence of the stdlib-encoder path with the harness' cached-embedding path."""
import sys, time, pickle, resource, importlib.util, statistics, os
zipdir, budget = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, 'dev'); import harness as H
eps = pickle.load(open('work/episodes3/S10.pkl', 'rb'))
ep = max(eps, key=lambda e: (len(e['pairs']), sum(len(e['items'][i]['item_content']) for i in e['items'])))
t0 = time.time()
spec = importlib.util.spec_from_file_location('m', os.path.join(zipdir, 'model.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.predict(H.inp(ep, ep['pairs'][0], ep['pairs'][0]['eval'][0][0], ''), [])   # loads artifacts
t_load = time.time() - t0
L = H.labeled_at(ep, budget)
targets = [H.inp(ep, p, ii, inter) for p in ep['pairs'] for (ii, inter, n, k) in p['eval']]
lat, preds = [], []
for t in targets:
    a = time.perf_counter(); preds.append(m.predict(t, L)); lat.append(time.perf_counter() - a)
print(f"budget={budget} labeled={len(L)} pairs={len(ep['pairs'])} targets={len(targets)} bench={ep['bench']}")
print(f"import+artifact load {t_load:.2f}s | first call (fit + encode) {lat[0]*1e3:.0f} ms | median call {statistics.median(lat)*1e3:.2f} ms | p99 {sorted(lat)[int(.99*len(lat))]*1e3:.1f} ms | max {max(lat)*1e3:.0f} ms | total {sum(lat):.2f}s")
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB")
# equivalence: same Predictor with the harness' numpy-cached embeddings instead of the stdlib encoder
EMB = pickle.load(open('work/emb_episode.pkl', 'rb'))
meta, arr, calib = m._load_art(os.path.join(zipdir, 'artifacts'))
P = m.Predictor(meta, arr, prior_sd=meta.get('prior_sd'), opts=meta.get('opts'), calib=calib, embed=lambda it: EMB[(it['item_content'], it['item_features'])].tolist())
d = max(abs(P.predict(t, L) - p) for t, p in zip(targets[:300], preds[:300]))
print(f"max |p(stdlib encoder) - p(cached numpy embeddings)| over 300 targets: {d:.2e}")
