"""Stdlib static sentence encoder (model2vec potion-base-8M): BERT-uncased WordPiece lookup + mean pooling + L2 norm.

Artifacts (bundled): embed_vocab.txt (one token per line, line = id), embed_table.f16 (rows of DIM little-endian float16).
No torch / numpy / network needed; runs inside predict(). Validated against model2vec in dev3/check_embed.py.
"""
import math
import os
import re
import struct
import unicodedata

DIM = 256
MAX_TOKENS = 256
MAX_CHARS = 1800
_HERE = os.path.dirname(os.path.abspath(__file__))
_SPECIAL = re.compile(r"(\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\])")


def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _basic_tokens(text):
    """BertNormalizer(clean_text, handle_chinese_chars, lowercase, strip_accents) + BertPreTokenizer."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD:
            continue
        cat = unicodedata.category(ch)
        if ch in "\t\n\r" or cat == "Zs":
            out.append(" ")
        elif cat.startswith("C"):
            continue
        elif _is_cjk(cp):
            out.append(" " + ch + " ")
        else:
            out.append(ch)
    text = "".join(out).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    toks, cur = [], []
    for ch in text:
        if ch.isspace():
            if cur:
                toks.append("".join(cur)); cur = []
        elif _is_punct(ch):
            if cur:
                toks.append("".join(cur)); cur = []
            toks.append(ch)
        else:
            cur.append(ch)
    if cur:
        toks.append("".join(cur))
    return toks


class Encoder:
    def __init__(self, path=None):
        path = path or os.path.join(_HERE, "artifacts")
        with open(os.path.join(path, "embed_vocab.txt"), encoding="utf-8") as fh:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(fh)}
        with open(os.path.join(path, "embed_table.f16"), "rb") as fh:
            self.buf = fh.read()
        self.unk = self.vocab["[UNK]"]
        self._rows = {}
        self._wp = {}

    def _wordpiece(self, word):
        got = self._wp.get(word)
        if got is not None:
            return got
        if len(word) > 100:
            ids = [self.unk]
        else:
            ids, start = [], 0
            while start < len(word):
                end, cur = len(word), None
                while start < end:
                    piece = word[start:end]
                    if start > 0:
                        piece = "##" + piece
                    if piece in self.vocab:
                        cur = self.vocab[piece]
                        break
                    end -= 1
                if cur is None:
                    ids = [self.unk]
                    break
                ids.append(cur)
                start = end
        if len(self._wp) > 200000:
            self._wp.clear()
        self._wp[word] = ids
        return ids

    def token_ids(self, text):
        ids = []
        for k, part in enumerate(_SPECIAL.split(text[:MAX_CHARS])):
            if k % 2:  # literal special token, kept atomic by the reference tokenizer
                ids.append(self.vocab[part])
                continue
            for w in _basic_tokens(part):
                ids.extend(self._wordpiece(w))
        # model2vec drops [UNK] before truncating
        return [i for i in ids if i != self.unk][:MAX_TOKENS]

    def _row(self, i):
        r = self._rows.get(i)
        if r is None:
            r = struct.unpack_from("<%de" % DIM, self.buf, i * DIM * 2)
            if len(self._rows) > 40000:
                self._rows.clear()
            self._rows[i] = r
        return r

    def encode(self, text):
        """-> list[float] of length DIM, L2-normalised (zero vector for empty text)."""
        ids = self.token_ids(text)
        if not ids:
            return [0.0] * DIM
        acc = [0.0] * DIM
        for i in ids:
            acc = [a + b for a, b in zip(acc, self._row(i))]
        nrm = math.sqrt(sum(a * a for a in acc))
        if nrm < 1e-12:
            return [0.0] * DIM
        return [a / nrm for a in acc]


def item_text(item):
    return ((item.get("item_content") or "")[:MAX_CHARS] + " " + (item.get("item_features") or "")[:300]).strip()
