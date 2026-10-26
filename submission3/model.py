"""Team Drift v3 predictor: cross-benchmark prior + test-time hierarchical logistic update with an item-embedding term + calibration.

logit p(s, i) = b + (1 + g) * zs(s) + zi(i) + beta + delta[pair(s)] + d[item(i)] + u . x(i)
  zs, zi : subject / item-text effects learned offline on the public training benchmarks (artifacts/ws, wi, idf)
  beta   : shift of the target benchmark (all revealed labels with the same benchmark_id)
  g      : slope on the subject-ability term for that benchmark
  delta  : subject x benchmark offset      d : item offset (revealed outcomes of other subjects on the same item)
  x(i)   : k-dim whitened PCA projection of the static text embedding of the item (embed.py, potion-base-8M, stdlib)
  u      : per-benchmark weights on x, MAP under N(0, su^2 I): lets revealed outcomes generalise to unrevealed items
beta, g, delta, d, u are MAP estimates under Gaussian priors, refitted from `labeled` on every distinct evidence set.
Calibration (artifacts/calib.json): a per-label-budget logit map (Platt) or isotonic map fitted out-of-fold, applied last.
With empty `labeled` and no calibration file this is the offline prior. Pure standard library.
"""
import array
import bisect
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import featurize as F  # noqa: E402

PRIOR_SD = {"beta": 1.5, "g": 0.35, "delta": 0.8, "d": 1.5}
OPTS = {"su": 0.3, "k": 16}
CLIP = (0.01, 0.99)
SWEEPS = 12
EMB_DIM = 256


def bucket_of(n_pair):
    """Label-budget bucket of the target's own subject-benchmark pair (budgets 0,1,3,7,15,31 map to 0..5)."""
    return 0 if n_pair <= 0 else 1 if n_pair == 1 else 2 if n_pair <= 3 else 3 if n_pair <= 7 else 4 if n_pair <= 15 else 5


def _f32(path):
    a = array.array("f")
    with open(path, "rb") as fh:
        a.frombytes(fh.read())
    return a


def _load_art(path):
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    arr = {name: _f32(os.path.join(path, name + ".f32")) for name in ("ws", "wi", "idf")}
    if os.path.exists(os.path.join(path, "pca_basis.f32")):
        arr["pca_mean"] = _f32(os.path.join(path, "pca_mean.f32"))
        arr["pca_basis"] = _f32(os.path.join(path, "pca_basis.f32"))  # kmax x EMB_DIM, rows already scaled to unit variance
    calib = None
    if os.path.exists(os.path.join(path, "calib.json")):
        with open(os.path.join(path, "calib.json")) as fh:
            calib = json.load(fh)
    return meta, arr, calib


def _sig(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _chol_solve(H, g):
    """Solve H x = g for symmetric positive-definite H (list of lists), Cholesky; H is not modified."""
    n = len(g)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = H[i][j] - sum(L[i][m] * L[j][m] for m in range(j))
            L[i][j] = math.sqrt(max(s, 1e-9)) if i == j else s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (g[i] - sum(L[i][m] * y[m] for m in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[m][i] * x[m] for m in range(i + 1, n))) / L[i][i]
    return x


class Predictor:
    def __init__(self, meta, arr, prior_sd=None, clip=CLIP, temp=1.0, opts=None, calib=None, embed=None):
        self.b, self.ws, self.wi, self.idf = meta["b"], arr["ws"], arr["wi"], arr["idf"]
        self.sd = dict(PRIOR_SD, **(prior_sd or {}))
        self.clip, self.temp, self.calib = clip, temp, calib
        self.opts = dict(OPTS, **(opts or {}))
        self.k = int(self.opts["k"]) if "pca_basis" in arr else 0
        self.mean = arr.get("pca_mean")
        self.basis = arr.get("pca_basis")
        self.embed = embed  # callable item dict -> list[float] (EMB_DIM) ; injected by the harness for caching
        self._enc = None
        self._fit_cache, self._item_cache, self._subj_cache, self._x_cache = {}, {}, {}, {}

    # ---- offline prior -------------------------------------------------
    def zi(self, item):
        key = (item.get("item_content"), item.get("item_features"), item.get("interactors"))
        v = self._item_cache.get(key)
        if v is None:
            tf = F.item_tf(item)
            ws = {j: c * self.idf[j] for j, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in ws.values())) + 1e-9
            v = sum(x * self.wi[j] for j, x in ws.items()) / nrm
            if len(self._item_cache) > 20000:
                self._item_cache.clear()
            self._item_cache[key] = v
        return v

    def zs(self, subject):
        key = tuple(sorted(subject.items()))
        v = self._subj_cache.get(key)
        if v is None:
            v = sum(self.ws[j] for j in F.subject_idx(subject))
            self._subj_cache[key] = v
        return v

    def xvec(self, item):
        """Whitened k-dim embedding features of an item (cached by item text)."""
        key = (item.get("item_content"), item.get("item_features"))
        v = self._x_cache.get(key)
        if v is None:
            if self.embed is not None:
                e = self.embed(item)
            else:
                if self._enc is None:
                    import embed as E
                    self._enc = E.Encoder(os.path.join(_HERE, "artifacts"))
                    self._etext = E.item_text
                e = self._enc.encode(self._etext(item))
            m, B, k = self.mean, self.basis, self.k
            c = [e[j] - m[j] for j in range(EMB_DIM)]
            v = [sum(B[r * EMB_DIM + j] * c[j] for j in range(EMB_DIM)) for r in range(k)]
            if len(self._x_cache) > 20000:
                self._x_cache.clear()
            self._x_cache[key] = v
        return v

    # ---- test-time MAP fit --------------------------------------------
    def _fit(self, rows):
        """rows: list of (pair_key, item_key, zs, rest, y, x). Returns (beta, g, delta{}, d{}, u[])."""
        sd, k = self.sd, self.k
        beta = g = 0.0
        delta = {r[0]: 0.0 for r in rows}
        d = {r[1]: 0.0 for r in rows}
        u = [0.0] * k
        by_pair, by_item = {}, {}
        for n, r in enumerate(rows):
            by_pair.setdefault(r[0], []).append(n)
            by_item.setdefault(r[1], []).append(n)
        eta = [r[3] + r[2] for r in rows]  # logit with beta=g=delta=d=u=0 (rest includes b+zi)
        iv = {key: 1.0 / (sd[key] ** 2) for key in sd}
        iu = 1.0 / (self.opts["su"] ** 2) if k else 0.0
        N = len(rows)

        def step(idx, cur, prior_prec, wfun=None):
            gr = prior_prec * (-cur)
            h = prior_prec
            for n in idx:
                p = _sig(eta[n])
                x = 1.0 if wfun is None else wfun(n)
                gr += x * (rows[n][4] - p)
                h += x * x * p * (1.0 - p)
            return max(-1.5, min(1.5, gr / h))

        for _ in range(SWEEPS):
            st = step(range(N), beta, iv["beta"])
            beta += st
            for n in range(N):
                eta[n] += st
            st = step(range(N), g, iv["g"], lambda n: rows[n][2])
            g += st
            for n in range(N):
                eta[n] += st * rows[n][2]
            for key, idx in by_pair.items():
                st = step(idx, delta[key], iv["delta"])
                delta[key] += st
                for n in idx:
                    eta[n] += st
            for key, idx in by_item.items():
                st = step(idx, d[key], iv["d"])
                d[key] += st
                for n in idx:
                    eta[n] += st
            if k:
                gr = [-iu * u[a] for a in range(k)]
                H = [[iu if a == b else 0.0 for b in range(k)] for a in range(k)]
                for n in range(N):
                    p = _sig(eta[n])
                    w, x, r = p * (1.0 - p), rows[n][5], rows[n][4] - p
                    for a in range(k):
                        gr[a] += x[a] * r
                        wa = w * x[a]
                        Ha = H[a]
                        for b in range(a + 1):
                            Ha[b] += wa * x[b]
                for a in range(k):
                    for b in range(a):
                        H[b][a] = H[a][b]
                du = _chol_solve(H, gr)
                nrm = math.sqrt(sum(t * t for t in du))
                if nrm > 1.5:
                    du = [t * 1.5 / nrm for t in du]
                for a in range(k):
                    u[a] += du[a]
                for n in range(N):
                    x = rows[n][5]
                    eta[n] += sum(du[a] * x[a] for a in range(k))
        return beta, g, delta, d, u

    def _evidence(self, labeled, bid):
        rows = []
        for (subject, item), y in labeled:
            if item.get("benchmark_id") != bid or y not in (0, 1):
                continue
            pk = tuple(sorted(subject.items()))
            ik = (item.get("item_content"), item.get("item_features"))
            rows.append((pk, ik, self.zs(subject), self.b + self.zi(item), int(y), self.xvec(item) if self.k else None))
        return rows

    def predict_logit(self, target, labeled):
        """-> (logit, number of revealed labels for the target's own subject-benchmark pair)."""
        subject, item = target
        z_s, z_i = self.zs(subject), self.zi(item)
        z = self.b + z_s + z_i
        n_pair = 0
        if labeled:
            bid = item.get("benchmark_id")
            fp = (bid, len(labeled), hash(tuple((e[0][1].get("item_content", ""), e[0][1].get("item_features", ""),
                                                  tuple(sorted(e[0][0].items())), e[1]) for e in labeled)))
            fit = self._fit_cache.get(fp)
            if fit is None:
                rows = self._evidence(labeled, bid)
                fit = self._fit(rows) if rows else (0.0, 0.0, {}, {}, [0.0] * self.k)
                if len(self._fit_cache) > 8:
                    self._fit_cache.clear()
                self._fit_cache[fp] = fit
            beta, gam, delta, d, u = fit
            pk = tuple(sorted(subject.items()))
            ik = (item.get("item_content"), item.get("item_features"))
            z += beta + gam * z_s + delta.get(pk, 0.0) + d.get(ik, 0.0)
            if self.k:
                x = self.xvec(item)
                z += sum(u[a] * x[a] for a in range(self.k))
            n_pair = sum(1 for (s, it), y in labeled if it.get("benchmark_id") == bid and s == subject)
        return z, n_pair

    def calibrate(self, z, n_pair):
        c = self.calib
        if not c or c.get("kind") == "none":
            return _sig(z / self.temp)
        cb = c["buckets"][bucket_of(n_pair)]
        if c["kind"] == "platt":
            return _sig(cb["a"] * z + cb["c"])
        p = _sig(z)  # isotonic: piecewise-linear map on the raw probability
        xs, ys = cb["x"], cb["y"]
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, p)
        t = (p - xs[i - 1]) / max(xs[i] - xs[i - 1], 1e-12)
        return ys[i - 1] + t * (ys[i] - ys[i - 1])

    def predict(self, target, labeled):
        z, n_pair = self.predict_logit(target, labeled)
        return min(self.clip[1], max(self.clip[0], self.calibrate(z, n_pair)))


_P = None


def predict(input, labeled=None):
    global _P
    if _P is None:
        meta, arr, calib = _load_art(os.path.join(_HERE, "artifacts"))
        _P = Predictor(meta, arr, prior_sd=meta.get("prior_sd"), opts=meta.get("opts"), calib=calib)
    return float(_P.predict(input, labeled or []))
