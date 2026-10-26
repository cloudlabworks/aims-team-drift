"""Time the shipped predict() through its public interface in a fresh process, from an extracted zip dir."""
import sys, time, pickle, resource, importlib.util, statistics, os
zipdir, budget = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, 'dev'); import harness as H
eps = pickle.load(open('work/episodes/S10.pkl', 'rb'))
ep = max(eps, key=lambda e: (len(e['pairs']), sum(len(e['items'][i]['item_content']) for i in e['items'])))
t0 = time.time()
spec = importlib.util.spec_from_file_location('m', os.path.join(zipdir, 'model.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.predict(H.inp(ep, ep['pairs'][0], ep['pairs'][0]['eval'][0][0], ''), [])   # loads artifacts
t_load = time.time() - t0
L = H.labeled_at(ep, budget)
targets = [H.inp(ep, p, ii, inter) for p in ep['pairs'] for (ii, inter, n, k) in p['eval']]
lat = []
for t in targets:
    a = time.perf_counter(); m.predict(t, L); lat.append(time.perf_counter() - a)
print(f"budget={budget} labeled={len(L)} pairs={len(ep['pairs'])} targets={len(targets)} bench={ep['bench']}")
print(f"import+artifact load {t_load:.2f}s | first call (fit) {lat[0]*1e3:.1f} ms | median call {statistics.median(lat)*1e3:.2f} ms | p99 {sorted(lat)[int(.99*len(lat))]*1e3:.1f} ms | max {max(lat)*1e3:.0f} ms | total {sum(lat):.2f}s")
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB")
