"""v3 unseen-subject split: subject GROUPS (= model identity) and the diagonal fold design.

Assumption (see work/PLAN.md, "v3"): the hidden test is a benchmark cold start AND the platform picks the subjects, so a test
subject may never have been seen in training. A model identity is the casefolded `normalized_name` (all harness / effort /
version variants of one model share a group); where normalized_name is missing we fall back to the casefolded `display_name`
from subjects.parquet (cache subject order == sorted subject_id, see dev/prep.py).

Folds: diagonal. Fold f holds out benchmark-fold f (dev/folds.py) AND subject-group-fold f. The prior for fold f is trained on
benchmarks outside f using only subjects outside f; its episodes are built on benchmarks in f from subjects in f only.
Subject groups are stratified by how many benchmarks they appear in so that one popular model cannot dominate a fold.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = 'data/measurement-db'
OUT = 'work/subject_groups.json'


def _clean(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip().casefold()


def group_of(row):
    return _clean(row['normalized_name']) or ('dn:' + _clean(row['display_name'])) or 'unknown'


def build(benches, K=5, seed=0):
    per_bench, appear = {}, {}
    cat = json.load(open('work/catalog.json'))
    for b in benches:
        p = f'{ROOT}/{b}/formatted_tables/'
        r = pd.read_parquet(p + 'responses.parquet', columns=['subject_id', 'response'])
        # identical to dev/prep.py: cache subject index = rank in sorted unique subject_id among rows with a response
        su = sorted(r[r.response.notna()].subject_id.unique())
        s = pd.read_parquet(p + 'subjects.parquet').set_index('subject_id').loc[su].reset_index()
        cp = f'work/cache/{b}.pkl'
        if os.path.getsize(cp) < 60_000_000:  # alignment check against the real cache for every benchmark small enough to load
            import pickle
            d = pickle.load(open(cp, 'rb'))
            names = [('' if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x))[:400] for x in s.normalized_name]
            assert len(d['subjects']) == len(s) and [x['normalized_name'] for x in d['subjects']] == names, b
        per_bench[b] = [group_of(row) for _, row in s.iterrows()]
        for g in set(per_bench[b]):
            appear[g] = appear.get(g, 0) + 1
    rng = np.random.default_rng(seed)
    # stratify: sort groups by #benchmarks (desc), random tie-break, deal round-robin in a snake pattern
    names = sorted(appear, key=lambda g: (-appear[g], rng.random()))
    fold_of, order = {}, list(range(K)) + list(range(K - 1, -1, -1))
    for j, g in enumerate(names):
        fold_of[g] = order[j % (2 * K)]
    return per_bench, fold_of, appear


if __name__ == '__main__':
    rel = json.load(open('work/release.json'))
    benches = sorted(f[:-4] for f in os.listdir('work/cache') if rel.get(f[:-4]) == 'public')
    per_bench, fold_of, appear = build(benches)
    json.dump(dict(per_bench=per_bench, fold_of=fold_of, appear=appear), open(OUT, 'w'))
    K = 5
    print('benchmarks', len(benches), 'groups', len(fold_of))
    for f in range(K):
        gs = [g for g, v in fold_of.items() if v == f]
        print('subject fold', f, 'groups', len(gs), 'sum appearances', sum(appear[g] for g in gs),
              'top', sorted(((appear[g], g) for g in gs), reverse=True)[:3])
