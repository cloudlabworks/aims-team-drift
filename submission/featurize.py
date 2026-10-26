"""Hashed text / subject featurisation shared by training and the submitted predictor (stdlib only)."""
import math
import re
import zlib

D_ITEM = 1 << 17
D_SUBJ = 1 << 15
_WORD = re.compile(r"[a-z0-9_]+")
_MAXTOK, _MAXBI = 300, 120


def _h(prefix, token, dim):
    return zlib.crc32((prefix + token).encode("utf-8", "ignore")) & (dim - 1)


def item_tf(item):
    """Item dict -> {hashed index: sublinear tf} (not idf-weighted, not normalised)."""
    cnt = {}

    def add(prefix, token):
        j = _h(prefix, token, D_ITEM)
        cnt[j] = cnt.get(j, 0) + 1

    content = (item.get("item_content") or "")[:3000]  # training caches were cut at 3000 chars
    text = content.lower()
    toks = _WORD.findall(text)[:_MAXTOK]
    for t in toks:
        add("w:", t)
    for a, b in zip(toks[:_MAXBI], toks[1:_MAXBI + 1]):
        add("b:", a + " " + b)
    n_chars = len(content)
    add("len:", str(int(math.log2(n_chars + 1))))
    add("ntok:", str(int(math.log2(len(toks) + 1))))
    feats = (item.get("item_features") or "")[:600]
    for piece in re.split(r"[;|]", feats):
        piece = piece.strip().lower()
        if piece:
            add("f:", piece[:60])
            for t in _WORD.findall(piece)[:6]:
                add("fw:", t)
    for t in _WORD.findall((item.get("interactors") or "")[:300].lower())[:20]:
        add("i:", t)
    return {j: 1.0 + math.log(c) for j, c in cnt.items()}


def subject_idx(subject):
    """Subject dict -> sorted list of hashed binary feature indices (several granularities)."""
    g = lambda k: (subject.get(k) or "").strip().lower()
    name, prov, harn, eff = g("normalized_name"), g("provider"), g("harness"), g("reasoning_effort")
    ver, rel, acc, extra = g("harness_version"), g("release_date"), g("access_date"), g("subject_features_extra")
    f = ["name=" + name, "prov=" + prov, "harn=" + harn, "eff=" + eff, "ver=" + harn + ":" + ver,
         "name+eff=" + name + "|" + eff, "name+harn=" + name + "|" + harn,
         "name+harn+eff=" + name + "|" + harn + "|" + eff, "year=" + rel[:4], "month=" + rel[:7],
         "accyear=" + acc[:4], "prov+year=" + prov + "|" + rel[:4], "harn+eff=" + harn + "|" + eff,
         "full=" + "|".join((name, prov, rel, acc, harn, ver, eff, extra[:200]))]
    if extra:
        for piece in re.split(r"[;|]", extra[:400])[:12]:
            piece = piece.strip()
            if piece and len(piece) < 80:
                f.append("x=" + piece)
        for t in _WORD.findall(extra[:200])[:12]:
            f.append("xw=" + t)
    for t in _WORD.findall(name)[:8]:
        f.append("nw=" + t)
    return sorted({_h("s:", s, D_SUBJ) for s in f})
