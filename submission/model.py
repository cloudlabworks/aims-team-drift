"""Team Drift predictor: cross-benchmark prior + test-time hierarchical logistic update (stdlib only).

logit p(s, i) = b + (1 + g) * zs(s) + zi(i) + beta + delta[pair(s)] + d[item(i)]
  zs, zi : subject / item-text effects learned offline on the public training benchmarks (artifacts/)
  beta   : shift of the target benchmark (all revealed labels with the same benchmark_id)
  g      : slope on the subject-ability term for that benchmark
  delta  : subject x benchmark offset      d : item offset (revealed outcomes of other subjects on the same item)
beta, g, delta, d are MAP estimates under Gaussian priors, refitted from `labeled` on every distinct evidence set.
With empty `labeled` the offline prior is returned unchanged.
"""
import array
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import featurize as F  # noqa: E402

PRIOR_SD = {"beta": 1.5, "g": 0.35, "delta": 0.8, "d": 1.5}
CLIP = (0.01, 0.99)
SWEEPS = 12


def _load_art(path):
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    arr = {}
    for name in ("ws", "wi", "idf"):
        a = array.array("f")
        with open(os.path.join(path, name + ".f32"), "rb") as fh:
            a.frombytes(fh.read())
        arr[name] = a
    return meta, arr


class Predictor:
    def __init__(self, meta, arr, prior_sd=None, clip=CLIP, temp=1.0, opts=None):
        self.b, self.ws, self.wi, self.idf = meta["b"], arr["ws"], arr["wi"], arr["idf"]
        self.sd = dict(PRIOR_SD, **(prior_sd or {}))
        self.clip, self.temp = clip, temp
        self.opts = dict(ss=1.0, si=1.0, var=0.0, du=0.5, v0=0.0)
        self.opts.update(opts or {})
        self._fit_cache = {}
        self._item_cache = {}
        self._subj_cache = {}

    # ---- offline prior -------------------------------------------------
    def zi(self, item):
        key = (item.get("item_content"), item.get("item_features"), item.get("interactors"))
        v = self._item_cache.get(key)
        if v is None:
            tf = F.item_tf(item)
            ws = {j: c * self.idf[j] for j, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in ws.values())) + 1e-9
            v = self.opts["si"] * sum(x * self.wi[j] for j, x in ws.items()) / nrm
            if len(self._item_cache) > 20000:
                self._item_cache.clear()
            self._item_cache[key] = v
        return v

    def zs(self, subject):
        key = tuple(sorted(subject.items()))
        v = self._subj_cache.get(key)
        if v is None:
            v = self.opts["ss"] * sum(self.ws[j] for j in F.subject_idx(subject))
            self._subj_cache[key] = v
        return v

    # ---- test-time MAP fit --------------------------------------------
    def _fit(self, rows):
        """rows: list of (pair_key, item_key, zs, rest, y). Returns (beta, g, delta{}, d{})."""
        sd = self.sd
        beta = g = 0.0
        delta = {r[0]: 0.0 for r in rows}
        d = {r[1]: 0.0 for r in rows}
        by_pair, by_item = {}, {}
        for n, r in enumerate(rows):
            by_pair.setdefault(r[0], []).append(n)
            by_item.setdefault(r[1], []).append(n)
        eta = [r[3] + r[2] for r in rows]  # logit with beta=g=delta=d=0 (rest includes b+zi)
        iv = {k: 1.0 / (sd[k] ** 2) for k in sd}

        def sig(x):
            return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

        def step(idx, cur, prior_prec, wfun=None):
            gr = prior_prec * (-cur)
            h = prior_prec
            for n in idx:
                p = sig(eta[n])
                x = 1.0 if wfun is None else wfun(n)
                gr += x * (rows[n][4] - p)
                h += x * x * p * (1.0 - p)
            return max(-1.5, min(1.5, gr / h))

        for _ in range(SWEEPS):
            st = step(range(len(rows)), beta, iv["beta"])
            beta += st
            for n in range(len(rows)):
                eta[n] += st
            st = step(range(len(rows)), g, iv["g"], lambda n: rows[n][2])
            g += st
            for n in range(len(rows)):
                eta[n] += st * rows[n][2]
            for k, idx in by_pair.items():
                st = step(idx, delta[k], iv["delta"])
                delta[k] += st
                for n in idx:
                    eta[n] += st
            for k, idx in by_item.items():
                st = step(idx, d[k], iv["d"])
                d[k] += st
                for n in idx:
                    eta[n] += st
        pn = [sig(e) * (1.0 - sig(e)) for e in eta]
        vb = 1.0 / (iv["beta"] + sum(pn))
        vp = {k: 1.0 / (iv["delta"] + sum(pn[n] for n in idx)) for k, idx in by_pair.items()}
        vd = {k: 1.0 / (iv["d"] + sum(pn[n] for n in idx)) for k, idx in by_item.items()}
        return beta, g, delta, d, (vb, vp, vd)

    def _evidence(self, labeled, bid):
        rows = []
        for (subject, item), y in labeled:
            if item.get("benchmark_id") != bid:
                continue
            pk = tuple(sorted(subject.items()))
            ik = (item.get("item_content"), item.get("item_features"))
            if y not in (0, 1):
                continue
            zs = self.zs(subject)
            rows.append((pk, ik, zs, self.b + self.zi(item), int(y)))
        return rows

    def predict_logit(self, target, labeled):
        subject, item = target
        z_s, z_i = self.zs(subject), self.zi(item)
        z = self.b + z_s + z_i
        if labeled:
            bid = item.get("benchmark_id")
            fp = (bid, len(labeled), hash(tuple((e[0][1].get("item_content", ""), e[0][1].get("item_features", ""),
                                                  tuple(sorted(e[0][0].items())), e[1]) for e in labeled)))
            fit = self._fit_cache.get(fp)
            if fit is None:
                rows = self._evidence(labeled, bid)
                fit = self._fit(rows) if rows else (0.0, 0.0, {}, {}, (self.sd["beta"] ** 2, {}, {}))
                if len(self._fit_cache) > 8:
                    self._fit_cache.clear()
                self._fit_cache[fp] = fit
            beta, gam, delta, d, (vb, vp, vd) = fit
            pk = tuple(sorted(subject.items()))
            ik = (item.get("item_content"), item.get("item_features"))
            z += beta + gam * z_s + delta.get(pk, 0.0) + d.get(ik, 0.0)
            if self.opts["var"]:
                v = vb + vp.get(pk, self.sd["delta"] ** 2) + vd.get(ik, self.opts["du"]) + self.opts["v0"]
                z = z / math.sqrt(1.0 + self.opts["var"] * (math.pi / 8.0) * v)
        return z

    def predict(self, target, labeled):
        z = self.predict_logit(target, labeled) / self.temp
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return min(self.clip[1], max(self.clip[0], p))


_P = None


def predict(input, labeled=None):
    global _P
    if _P is None:
        meta, arr = _load_art(os.path.join(_HERE, "artifacts"))
        _P = Predictor(meta, arr)
    return float(_P.predict(input, labeled or []))
