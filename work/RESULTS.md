# Team Drift — AIMS PAIEC predictor: local results (nothing submitted to Codabench)

Protocol facts and UNKNOWNs: see `work/PLAN.md`. All numbers are from our LOCAL harness (`dev/harness.py`), not from the organisers' scorer.

## What is shipped (`dist/submission.zip`, 838 KB: model.py, featurize.py, artifacts/{ws,wi,idf}.f32, meta.json; stdlib only)
logit p(s,i) = b + zs(s) + zi(i) + beta + g·zs + delta[pair] + d[item]
- **Offline prior** (zs, zi, b): weighted logistic ridge across the 167 public binary benchmarks; zs = hashed subject-attribute features (name, provider, harness, effort, year, ...), zi = hashed TF-IDF of item text/features. Benchmark weight ∝ (#trials)^0.5.
- **Test-time update**: from every `labeled` entry with the target's `benchmark_id` (all subjects pooled): MAP of benchmark shift beta (sd 1.5), subject-ability slope g (0.35), pair offset delta (0.8), item offset d (1.5; repeated items across subjects), Newton coordinate sweeps, memoised per labeled set. Budget 0: prior only. Output clipped to [0.01, 0.99].
- No `labeling.py` (platform's random acquisition).

## Local validation design (benchmark-level cold start)
- 167 public binary benchmarks (170 binary in the db minus 3 `release: withheld`: imagenet_hard, swe_together, visual_memory; 111 graded/non-binary benchmarks unused: organisers' binarisation is UNKNOWN). 151 of them have >=1 subject with >=80 items.
- Grouped 5-fold CV over BENCHMARKS (fold sizes 33/34/34/34/32); prior trained only on the other folds; held-out benchmark never seen in training.
- Episode = one held-out benchmark, S sampled subjects (pairs with >=80 items), item pool shared across subjects (cap: S × pool <= 1000 unique subject–item pairs, the formative cap), per-pair 50/50 acquisition/evaluation split, random acquisition (kit's `uniform < labels_remaining/items_remaining` rule, rng instead of sha256), nested budgets 0/1/3/7/15/31, labeled = union over pairs of their first n labels, all values strings, benchmark_id hashed, `interactors` from the response row. Repeats count as separate Brier targets with one prediction. ALC = 0.1/0.2/0.2/0.2/0.2/0.1 weights; pairs weighted equally.
- Settings: **S=10** (844 pairs, 53.5k eval targets), **S=3** (347 pairs), **S=10i** = 10 subjects with INDEPENDENT item pools (no item overlap helps cross-subject item effects; 1161 pairs).
- Sanity checks passed: const 0.5 = 0.2500 at every budget; empirical-mean B1 = 0.367 (>0.25, un-smoothed single label); stdlib baseline B0 = 0.25.
- Cells: mean ± std over the 5 benchmark folds (pairs pooled within fold).

## Main table (lower is better)
### S=10 (subjects per benchmark); cells = mean ± std over 5 benchmark-folds
| model | B0 | B1 | B3 | B7 | B15 | B31 | **ALC** | pairs |
|---|---|---|---|---|---|---|---|---|
| const05 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | **0.2500±0.0000** | 844 |
| empirical_mean | 0.250±0.000 | 0.367±0.025 | 0.244±0.014 | 0.210±0.007 | 0.196±0.008 | 0.189±0.009 | **0.2475±0.0115** | 844 |
| stdlib_baseline | 0.250±0.000 | 0.243±0.013 | 0.220±0.013 | 0.206±0.008 | 0.195±0.008 | 0.189±0.009 | **0.2166±0.0089** | 844 |
| final | 0.247±0.007 | 0.215±0.011 | 0.199±0.010 | 0.188±0.008 | 0.173±0.006 | 0.157±0.006 | **0.1953±0.0071** | 844 |
### S=3 (subjects per benchmark); cells = mean ± std over 5 benchmark-folds
| model | B0 | B1 | B3 | B7 | B15 | B31 | **ALC** | pairs |
|---|---|---|---|---|---|---|---|---|
| const05 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | **0.2500±0.0000** | 347 |
| empirical_mean | 0.250±0.000 | 0.366±0.018 | 0.242±0.011 | 0.211±0.007 | 0.199±0.008 | 0.193±0.010 | **0.2480±0.0079** | 347 |
| stdlib_baseline | 0.250±0.000 | 0.275±0.014 | 0.225±0.010 | 0.208±0.008 | 0.198±0.008 | 0.193±0.010 | **0.2254±0.0069** | 347 |
| final | 0.250±0.007 | 0.226±0.008 | 0.209±0.010 | 0.200±0.008 | 0.194±0.009 | 0.187±0.010 | **0.2094±0.0067** | 347 |
### S=10i (subjects per benchmark); cells = mean ± std over 5 benchmark-folds
| model | B0 | B1 | B3 | B7 | B15 | B31 | **ALC** | pairs |
|---|---|---|---|---|---|---|---|---|
| const05 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | 0.250±0.000 | **0.2500±0.0000** | 1161 |
| empirical_mean | 0.250±0.000 | 0.354±0.019 | 0.240±0.007 | 0.209±0.010 | 0.196±0.008 | 0.190±0.009 | **0.2436±0.0088** | 1161 |
| stdlib_baseline | 0.250±0.000 | 0.231±0.013 | 0.216±0.007 | 0.204±0.010 | 0.195±0.008 | 0.190±0.009 | **0.2131±0.0084** | 1161 |
| final | 0.250±0.007 | 0.211±0.013 | 0.201±0.010 | 0.194±0.010 | 0.187±0.009 | 0.182±0.009 | **0.2020±0.0087** | 1161 |

Paired ΔALC (final − baseline, same folds): `logs/paired_deltas.txt`
```
S=10: final - empirical_mean: per-fold dALC [-0.0606, -0.0487, -0.0478, -0.0548, -0.049] mean -0.0522 (negative on 5/5 folds)
S=10: final - stdlib_baseline: per-fold dALC [-0.0246, -0.0201, -0.0211, -0.0221, -0.0187] mean -0.0213 (negative on 5/5 folds)
S=3: final - empirical_mean: per-fold dALC [-0.0397, -0.0356, -0.0353, -0.043, -0.0395] mean -0.0386 (negative on 5/5 folds)
S=3: final - stdlib_baseline: per-fold dALC [-0.0168, -0.0155, -0.0135, -0.0177, -0.0169] mean -0.0161 (negative on 5/5 folds)
S=10i: final - empirical_mean: per-fold dALC [-0.0456, -0.0452, -0.0383, -0.0379, -0.0413] mean -0.0417 (negative on 5/5 folds)
S=10i: final - stdlib_baseline: per-fold dALC [-0.012, -0.0119, -0.0107, -0.0096, -0.0112] mean -0.0111 (negative on 5/5 folds)
```
Rows: `const05` = always 0.5; `empirical_mean` = the organisers' kit `empirical_mean/model.py` run unchanged (labels filtered by subject equality + benchmark_id); `stdlib_baseline` = the earlier pooled additive-logit/shrunk-mean model (`baselines/stdlib_model.py`); `final` = shipped model. The BLE baselines need API credentials and were NOT run. Organisers' published scores: UNKNOWN (the page shows only "illustrative" values).

## Ablations / sweeps (S=10, ΔALC vs shipped 0.1953; run with the prior trained on 170 benchmarks incl. the 3 withheld sets, aggregated on the 167 retained)
| variant | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ΔALC vs shipped |
|---|---|---|---|---|---|---|---|---|
| test-time d-prior sd 1.0 (earlier default) | 0.2474 | 0.2152 | 0.2002 | 0.1899 | 0.1773 | 0.1624 | 0.1975 | +0.0022 |
| d sd 0.5 | 0.2474 | 0.2161 | 0.2025 | 0.1947 | 0.1862 | 0.1768 | 0.2023 | +0.0070 |
| d sd 1.5 (shipped) | 0.2474 | 0.2148 | 0.1991 | 0.1876 | 0.1731 | 0.1566 | 0.1953 | +0.0000 |
| d sd 2.5 | 0.2474 | 0.2159 | 0.2012 | 0.1905 | 0.1756 | 0.1578 | 0.1971 | +0.0018 |
| beta sd 1.0 | 0.2474 | 0.2151 | 0.2001 | 0.1898 | 0.1773 | 0.1624 | 0.1974 | +0.0021 |
| beta sd 2.5 | 0.2474 | 0.2186 | 0.2011 | 0.1901 | 0.1773 | 0.1625 | 0.1984 | +0.0031 |
| pair-offset sd 0.4 | 0.2474 | 0.2149 | 0.2004 | 0.1908 | 0.1788 | 0.1642 | 0.1981 | +0.0028 |
| pair-offset sd 1.5 | 0.2474 | 0.2244 | 0.2075 | 0.1942 | 0.1792 | 0.1628 | 0.2021 | +0.0068 |
| slope sd 0.15 | 0.2474 | 0.2152 | 0.2002 | 0.1899 | 0.1773 | 0.1624 | 0.1975 | +0.0022 |
| slope sd 0.7 | 0.2474 | 0.2153 | 0.2002 | 0.1899 | 0.1773 | 0.1624 | 0.1975 | +0.0022 |
| temperature 0.9 | 0.2479 | 0.2164 | 0.2011 | 0.1906 | 0.1772 | 0.1616 | 0.1980 | +0.0027 |
| temperature 1.15 | 0.2469 | 0.2145 | 0.2000 | 0.1900 | 0.1783 | 0.1644 | 0.1977 | +0.0024 |
| per-benchmark evidence grid over (d, pair) sds | 0.2474 | 0.2154 | 0.2002 | 0.1903 | 0.1737 | 0.1555 | 0.1962 | +0.0009 |
| probit variance shrinkage du=0.5 | 0.2474 | 0.2145 | 0.1994 | 0.1882 | 0.1750 | 0.1592 | 0.1961 | +0.0008 |
| ... half strength | 0.2474 | 0.2142 | 0.1989 | 0.1876 | 0.1738 | 0.1577 | 0.1954 | +0.0001 |
| + text-kernel residual smoothing c=3 | 0.2474 | 0.2146 | 0.1984 | 0.1866 | 0.1718 | 0.1553 | 0.1945 | -0.0008 |
| + kernel c=6 | 0.2474 | 0.2157 | 0.2004 | 0.1896 | 0.1748 | 0.1576 | 0.1966 | +0.0013 |
| NO subject prior (zs=0) | 0.2587 | 0.2156 | 0.1995 | 0.1876 | 0.1732 | 0.1567 | 0.1967 | +0.0014 |
| NO item-text prior (zi=0) | 0.2477 | 0.2149 | 0.1992 | 0.1876 | 0.1732 | 0.1566 | 0.1954 | +0.0001 |
| subject prior x1.5 | 0.2529 | 0.2155 | 0.1993 | 0.1877 | 0.1732 | 0.1566 | 0.1961 | +0.0008 |
| prior lam_i 3e-3 | 0.2469 | 0.2148 | 0.1991 | 0.1875 | 0.1731 | 0.1566 | 0.1952 | -0.0001 |
| prior lam_i 1e-1 | 0.2476 | 0.2149 | 0.1992 | 0.1876 | 0.1731 | 0.1566 | 0.1954 | +0.0001 |
| prior lam_s 1e-3 | 0.2470 | 0.2150 | 0.1988 | 0.1873 | 0.1730 | 0.1566 | 0.1952 | -0.0002 |
| prior with benchmark intercepts | 0.2537 | 0.2158 | 0.1996 | 0.1877 | 0.1731 | 0.1567 | 0.1963 | +0.0010 |
| weak reg lam_i 1e-5 + bench intercepts | 0.2529 | 0.2145 | 0.1983 | 0.1868 | 0.1726 | 0.1565 | 0.1954 | +0.0001 |
| weak reg lam_i 1e-6, lam_s 1e-4 | 0.2565 | 0.2187 | 0.2003 | 0.1882 | 0.1737 | 0.1572 | 0.1976 | +0.0023 |

Reading: (1) the test-time hierarchical update is the whole gain; the item-text prior contributes nothing (zi=0 identical) — a NEGATIVE result for hashed-TF-IDF text transfer across benchmarks; weakly regularised text weights make B0 worse (0.253–0.257). (2) The subject prior only matters at B0 (0.259 -> 0.247). (3) Item-offset sd matters most (0.5 -> 1.5: 0.202 -> 0.195); per-benchmark evidence selection, probit variance shrinkage, and text-kernel smoothing did not beat the fixed setting by a meaningful margin. (4) Test-time sds were tuned on the S=10 evaluation folds (spread ~0.002 ALC vs a 0.021 paired gain over the stdlib baseline), so the S=10 number is slightly optimistic; S=3 and S=10i used the same fixed hyper-parameters.

## Runtime (`logs/runtime_v2.txt`, extracted zip, fresh process, M-series CPU, single thread)
Import+artifact load < 10 ms; 500 targets at budget 31 (310 labeled, 10 pairs): first call (MAP fit) 4.2 ms, then ~0.1 ms/call, peak RSS 86 MB. Worst case (310 labeled items of 108k chars): 0.04 s for the first call. Platform per-call timeout / memory limits are UNKNOWN; far below any plausible limit.

## Checks (literal output in `logs/check_final.txt`, `logs/unit_tests_v2.txt`)
```
$ cd paiec_baseline && ../.venv/bin/python -I check_submission_zip.py ../dist/submission.zip
OK: ../dist/submission.zip looks like a valid submission ZIP.
$ ... --static-only
OK: ../dist/submission.zip looks like a valid submission ZIP. Static checks only; prediction and acquisition were not run.
$ PATH=<venv>/bin:$PATH python -m unittest discover -s tools -p 'test_*.py'   # organisers' tests
Ran 19 tests ... OK
$ python3 -m unittest ... (system python 3.9 first on PATH): FAILED (failures=4, errors=2) - subprocess tests need Python >=3.10 (`str | None`); environment issue, not the submission.
own tests: tests/test_model.py: Ran 7 tests ... OK
```
Not run: the Codabench kit's run_smoke_test.py (not available offline), the streaming-coordinator path end to end, BLE variants, any Codabench submission.

## Limitations / caveats
- Harness reveal rule is equal in distribution, not identical, to the kit's sha256 rule; how the platform samples subjects/pairs, and whether pairs share items, is UNKNOWN (S=10 vs S=10i bracket it).
- Benchmark families that are near-duplicates (bbq/helm_bbq, igakuqa/igakuqa119, matharena/matharena_platform, swe_*, medhelm_*) may sit in different folds; since the prior only affects B0 the leakage is bounded (~0.1 × B0 effect).
- Identity-level unseen-subject validation NOT run; ss0 bounds the cost at ~+0.012 on B0 (~0.001 ALC).
- Only binary benchmarks are used; test benchmarks with graded responses are binarised by the organisers in an unknown way.
- No acquisition function; test-time hyper-parameters tuned on eval folds.

## Reproduce
```
uv venv; uv pip install numpy pandas pyarrow scipy scikit-learn pyyaml huggingface_hub   # data/ = HF aims-foundations/measurement-db
python dev/catalog.py; python dev/prep.py; python dev/feats.py                # work/catalog.json, work/cache, work/feats
python -c "import sys;sys.path.insert(0,'dev');import harness as H;[H.build_all(3),H.build_all(10),H.build_all(10,indep=True)]"
python dev/run_baselines.py {const05,empirical_mean,stdlib_baseline} {3,10,10i}
python dev/eval_model.py --S 10 --tag final        # (also 3, 10i)  -> work/results/
python dev/train_final.py 1e-2 1e-2 0.5            # -> submission/artifacts
python build_zip.py                                # -> dist/submission.zip
```
