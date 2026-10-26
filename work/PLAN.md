# Team Drift — AIMS PAIEC predictor: protocol + plan

Sources: **[Page]** = https://aimslab.stanford.edu/competition (fetched 2026-10-12; hidden "Predictor fields" sections read from the raw HTML);
**[Kit]** = paiec_baseline repo (check_submission_zip.py, tools/streaming_ingestion.py, empirical_mean/); **[API]** = aims/codabench_api.json;
**[DB]** = data/measurement-db/README.md. UNKNOWN = not stated in any of these.

## 1. Scoring metric
- Per label budget n in {0,1,3,7,15,31}: Brier B_n = mean (p - y)^2 over the evaluation-pool responses of one subject–benchmark pair. 0.5 constant = 0.25. [Page: Evaluation > Metric]
- Brier ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31 (trapezoid on x=log2(n+1), normalised). [Page: Computing Brier ALC]
- ALC is computed **per subject–benchmark pair, then averaged equally over pairs** (pairs weigh the same regardless of size). [Page]
- All recorded responses count in Brier/ECE ("a subject that attempted an item three times contributes three separate targets"); "Number of Unique Subject-Item Pairs" is only a count. Repeats with identical inputs reuse one prediction. [Page: Overview, Formative feedback, Predictor Interface]
- ECE (10 equal-width bins) is a diagnostic only, not scored. [Page]
- Final ranking: automatic Brier ALC on a common hidden test subset + manual grading of codebase and report (reproducibility is rerun by organisers). [Page: Summative evaluation]

## 2. Evaluation protocol
- Benchmark-level cold start: held-out items come from benchmarks in a private test pool; training pool = benchmarks in measurement-db. Pool assignment is fixed, collection may grow. [Page: Overview, Data]
- The platform selects the subjects; subject coverage may differ between pools (training subjects need not appear at test). subjects.txt is rejected by the checker. [Page; Kit check_submission_zip.py]
- Each formative evaluation samples a fresh subset of hidden data capped at **1,000 unique subject–item pairs across acquisition + evaluation pools**; summative uses one common subset. [Page: Data]
- For each subject–benchmark pair with **>=80 distinct items**, items are split randomly 50/50 into acquisition and evaluation pools, fixed across submissions and budgets; all responses for an item stay together. Acquisition items are never scored. [Page: Test-time Adaptation]
  UNKNOWN: what happens to pairs with <80 items (presumably budget 0 only or excluded).
- Budgets 0,1,3,7,15,31 per pair, nested labels from one acquisition trajectory, same evaluation items at every budget. [Page]
- Acquisition: candidates (acquisition-pool items) arrive one by one in a fixed random order; acquisition_function(input, prediction, labeled, context) returns True to reveal. Without labeling.py selection is random: the kit draws `uniform(sha256(context,input)) < min(1, labels_remaining/items_remaining)`. [Page; Kit streaming_ingestion.py L~150]
- At each budget the platform calls predict(input, labeled) once per distinct [subject, item]. [Page]

## 3. What predict(input, labeled) receives
- input = [subject, item]; every value a plain string, missing = "". [Page: Predictor fields]
  - subject keys (exactly): normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra.
  - item keys (exactly): item_content, item_features, interactors, benchmark_id (stable anonymous id, shared by targets and acquired examples of that benchmark).
  - NOTE: no benchmark metadata (name/description/domain) is passed, only the anonymous benchmark_id, so benchmark identity must be inferred from item text/features.
- labeled = list of [[subject, item], label] with label 0/1; "across sampled subject–benchmark pairs", i.e. it can include other subjects/benchmarks (kit's empirical_mean filters by subject equality + benchmark_id). Empty at budget 0. [Page; Kit empirical_mean/model.py]
- Return finite float in [0,1] (float() applied; NaN/inf/out of range rejected -> code 42). [Page: Predictor output]
- State: the kit runs evaluation workers recreated per checkpoint ("Evaluation workers are recreated with frozen evidence at each checkpoint") so nothing may be assumed to persist between calls; module-level caches are only an optimisation. [Kit streaming_ingestion.py docstring]

## 4. Limits
- 8 hours per run including setup (execution_time_limit 28800 s). [Page FAQ; API phases]
- Up to 5 HF model repos in models.txt: 300B params / 1000 GB each, 1024 GB combined; GPU allocation automatic; models cannot be downloaded at runtime; extra package installation and trust_remote_code off by default, requirements.txt "organizer policy gated". [Page]
- Network at scoring: approved inference APIs only; web search/browsing prohibited. [Page: Participation > Runtime]
- Exit codes: 10/11 invalid model.py, 12 invalid labeling.py, 40 prediction exception, 41 call timeout, 42 invalid probability, 50 total timeout. [Page]
- UNKNOWN: per-call timeout value, memory limit, CPU count, concurrency, which Python packages are preinstalled (docker image us-west1-docker.pkg.dev/soe-cs321m-competition/paiec/submission:latest, not inspected). BLE README mentions a 60 s embedding budget and 120 s response attempt, not the platform limit. => keep submission stdlib+numpy-optional, fall back to pure Python if numpy import fails.
- Bundled artifacts may be .json/.npz/.npy/.pkl/.parquet/.txt etc. ([Kit] SAFE_ARTIFACT_SUFFIXES); checker flags literal bundled-file paths that are missing; HF refs in from_pretrained/SentenceTransformer must be in models.txt or bundled.
- Disclosure: all training sources, pretrained models, LLM coding assistants in the report; presentation consideration needs released code + any non-measurement-db training data. [Page]

## 5. Published baselines
- empirical_mean (stdlib; 0.5 at budget 0, else un-smoothed pair mean), empirical_mean_ble_acquisition, BLE (LLM + retrieval over public data, needs API key). [Kit README]
- Their scores: **UNKNOWN**; no number is documented. The page's 0.140 / 0.22 are labelled illustrative. Kit README says empirical mean can score worse than constant 0.5 (unsmoothed). We measure them locally instead (section 6).

## 6. Local validation design (work/)
- Hold out whole benchmarks (grouped K-fold over benchmarks); models only see other benchmarks as training data.
- Pairs: each (subject, benchmark) with >=80 distinct items; items split 50/50 acq/eval by a hash; acquisition order = fixed hash order; budget k labels = first k revealed by the kit's random rule, nested.
- Metric: Brier per budget over all evaluation-item responses (repeats counted), ALC with weights above, mean over pairs; per-fold mean +- std over folds.
- Only benchmarks whose responses are binary (0/1, NaN dropped) are used for scoring; the organisers' binarisation of graded benchmarks is UNKNOWN.

## 7. Addendum (after building)
- Data: 281 benchmark dirs; 170 have purely binary responses; 3 of those are `release: withheld` per the db README (imagenet_hard, swe_together, visual_memory) and are excluded from training/eval => 167. 111 graded benchmarks unused. [DB README]
- Facts found while building: `labeled` may span several subjects/benchmarks (filter on benchmark_id; pooling other subjects' labels on the same benchmark is the main lever); the formative 1,000-pair cap implies only a few pairs per benchmark.
- Results: work/RESULTS.md.


## v3 (2026-10-13): unseen-subject validation, item embeddings, labeling.py, nested calibration

**Verdict: READY (by the pre-registered rule), with a small effect.** Primary comparison = paired ΔALC (v3 − v2) on the S=10 unseen-subject split, bootstrap over benchmark clusters: **−0.0011 [−0.0019, −0.0002]** (95% CI excludes zero; subject-group bootstrap [−0.0016, −0.0005]). Same sign and CI excluding zero on S=3 (−0.0012 [−0.0019, −0.0004]) and S=10i (−0.0013 [−0.0022, −0.0005]). It is ~0.5% relative; ECE did not improve (see below). No Codabench submission, account action or post was made. Deliverable: `dist/submission-v3.zip` (15,052,218 bytes, sha256 71fe0ee2…e85856, not committed; rebuild with `python build_zip.py submission3 dist/submission-v3.zip`). `submission3/artifacts/embed_table.f16` (15,118,336 bytes, git-ignored, rebuilt bit-identically by `dev3/make_embed_artifacts.py`) sha256 cd5b65b0005dd6c26c832abf041b15c925bd3dee49bf32eec57220306d070bb3.

### 1. Validation design (dev3/groups.py, harness3.py, priors.py, eval3.py)
- **Assumption (stated, not verified by the organisers' text):** the hidden test is a benchmark cold start (PLAN §2) and "the platform selects the subjects; training subjects need not appear at test" (§2), so a test subject may never have been seen in training. A *subject* is a model identity = casefolded `normalized_name` (all harness/effort/version variants of one model share a group; if missing, `display_name` from subjects.parquet). 4,189 groups over the 167 public binary benchmarks; alignment cache-subject-index ↔ subjects.parquet asserted for every benchmark whose cache is <60 MB.
- **Diagonal 5-fold design:** fold f holds out benchmark-fold f (same folds as v2) AND subject-group-fold f (groups dealt round-robin in a snake order sorted by #benchmarks they appear in, so GPT-4o-class models cannot dominate one fold). The prior for fold f is trained on benchmarks outside f using only rows of subjects outside f; fold-f episodes use only fold-f subjects. 5 outer priors + 20 inner priors (outside {f,j}); v2's hyper-parameters (lam 1e-2/1e-2, alpha 0.5) and test-time sds frozen. Artifacts were retrained (v2's old `fold*_.npz` saw all subjects and are used only as a labelled control).
- Episodes: 5 draws per benchmark (seeded), S ∈ {10 (shared item pool), 3, 10i (independent pools)} as in v2, but fewer pairs/episode (only fold-f subjects are eligible: ≈3.4 pairs per S=10 episode vs ≈10 in v2's split). Pairs: S=10 1,907 / S=3 1,170 / S=10i 2,645 over 113 benchmarks (benchmarks with no eligible held-out subject drop out). Acquisition and labels are produced by `submission3/labeling.py` (kit-exact sha256 rule; hash-based label draw) so an episode is a pure function of its seed.
- **Statistics:** pair-level (pairs weigh equally, as in the organisers' ALC). Paired bootstrap, 4,000 resamples, resampling whole **benchmarks** (primary; the test is a benchmark cold start) and, as a second view, whole **subject groups**; same resamples for all models. ECE = 10 equal-width bins, computed per pair and budget then averaged over pairs and the six budgets (organisers' pooling unknown; the response-pooled value is also printed and is dominated by a few pairs with millions of repeats). Sanity: const 0.5 = 0.2500 at every budget; kit empirical_mean B1 = 0.372 (>0.25).
- **Control:** v2 scored with its old subject-*seen* priors on the same episodes: ΔALC vs v2-unseen = +0.0001 [−0.0003, +0.0006] (S=10). So unseen-subject training costs v2 nothing measurable; v2's 0.195 → 0.206 move between the old and new tables is the different episode composition (fewer subjects per benchmark to pool), not leakage.

### 2. Results (lower is better; benchmark-cluster 95% CIs; full tables + subject-group bootstrap: `logs/analysis_v3_main.md`)
S=10 (primary), 1,907 pairs, 113 benchmarks, 474 subject groups
| model | B0 | B1 | B3 | B7 | B15 | B31 | ALC [95% CI] | ECE [95% CI] |
|---|---|---|---|---|---|---|---|---|
| const 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2057 |
| kit empirical_mean | 0.2500 | 0.3721 | 0.2505 | 0.2186 | 0.2047 | 0.1981 | 0.2540 [0.2392, 0.2674] | 0.1802 [0.1721, 0.1871] |
| v2 (retrained, unseen subjects) | 0.2488 | 0.2241 | 0.2085 | 0.1985 | 0.1870 | 0.1747 | 0.2060 [0.1950, 0.2163] | 0.1405 [0.1348, 0.1460] |
| v3, embeddings only | 0.2488 | 0.2247 | 0.2084 | 0.1977 | 0.1857 | 0.1731 | 0.2055 [0.1946, 0.2160] | 0.1440 [0.1382, 0.1496] |
| **v3 (embeddings + calibration)** | 0.2480 | 0.2233 | 0.2078 | 0.1972 | 0.1856 | 0.1731 | **0.2049** [0.1946, 0.2146] | 0.1435 [0.1378, 0.1491] |

Paired deltas (A − B), S=10: v3 − v2 ΔALC −0.0011 [−0.0019, −0.0002], P(Δ<0)=0.991; v3(emb only) − v2 −0.0005 [−0.0009, −0.0001]; v3 − kit −0.0491 [−0.0552, −0.0427]; v2 − kit −0.0480 [−0.0538, −0.0421]; ΔECE v3 − v2 +0.0030 [+0.0007, +0.0053] (slightly worse).

Other settings (ALC [CI]; v3 − v2 ΔALC [CI]): S=3: kit 0.2482 [0.2346, 0.2610], v2 0.2101 [0.1997, 0.2200], v3 0.2089 [0.1991, 0.2182], Δ −0.0012 [−0.0019, −0.0004]. S=10i: kit 0.2505 [0.2359, 0.2641], v2 0.2062 [0.1946, 0.2168], v3 0.2049 [0.1939, 0.2150], Δ −0.0013 [−0.0022, −0.0005].

Per-budget Brier with 95% CIs, paired per-budget ΔBrier and the response-pooled ECE with its CI are in `logs/analysis_v3_main.md`. S=10 v3 − v2 ΔBrier per budget is negative from B7 on (B7 −0.0013, B15 −0.0016, B31 −0.0016, all CIs excluding zero). **"brier" vs the Codabench field (0.17587): the organisers' aggregation is UNKNOWN** (not stated in the page text we hold); the closest local analogues are B31 (v2 0.1747, v3 0.1731) and the ALC (v2 0.2060, v3 0.2049).

**ECE caveat (read before uploading):** the page says all recorded responses count in Brier/ECE, which points to a response-pooled ECE. On that version v3 is *higher* (0.1669 vs v2 0.1585; paired Δ +0.0084 [−0.0145, +0.0088], dominated by a few pairs with millions of repeated responses; embeddings alone 0.1466). On the pair-mean version v3 is also slightly higher (+0.0030). So the formative `calibration_ece` (0.151 for v2) will probably not fall and may rise if v3 is uploaded, while Brier ALC improves by ~0.001: the pair-equal temperature shrinkage helps the typical pair and does not help the few huge low-accuracy pairs that dominate pooled ECE. ECE is not scored.

### 3. What was built
- **Item embeddings (`submission3/embed.py`):** model2vec `minishlab/potion-base-8M` (MIT, 256-d static embeddings), bundled as a float16 table + vocab and run by a **stdlib** WordPiece + mean-pool encoder, because the scoring image's packages are UNKNOWN and torch / sentence-transformers cannot be assumed (so no models.txt, no network, no extra packages). Validated against model2vec on 492 texts: mean cosine 0.99992, min 0.991 (`logs/embed_check_v3.txt`); shipped path vs harness' cached-embedding path: max |Δp| 8e-9. Used at test time: x(i) = whitened PCA(k) of the embedding; logit += u·x(i) with u a per-benchmark MAP (Gaussian prior, sd su) fitted in the same Newton sweeps as beta, g, delta, d, so revealed outcomes generalise to unrevealed items. The offline prior is unchanged (v2 ablation: item-text prior contributes nothing, so no embedding-prior variant was run).
- **Tuning (inner folds only; `logs/select_v3.txt`):** grid k ∈ {0,8,16,32} × su ∈ {0.05…1.0} × d-sd ∈ {1.5, 0.75} scored on out-of-fold fold-j episodes (j ≠ f, prior inner(f,j)), averaged over S ∈ {10,3,10i}; every outer fold independently chose **k=32, su=0.15, d-sd 1.5** (inner gain vs k=0: −0.0006 ALC; k=16, su=0.2 is within 0.0001 and half the compute; stronger su (≥0.5) hurts).
- **Calibration (`dev3/calibrate.py`, `logs/calibrate_v3.txt`):** per label-budget bucket (budgets 0/1/2-3/4-7/8-15/16+ of the target's own pair) maps fitted on out-of-fold raw logits of inner folds only (outer fold f's benchmarks AND subjects never seen), family chosen by cross-fitting over inner folds among none / temperature / Platt / isotonic: **temperature** won in all 5 folds (isotonic worst). Shipped temperatures (multiply the logit): B0 0.497, B1 0.78, B2-3 0.856, B4-7 0.884, B8-15 0.927, B16+ 0.958 (the offline prior is ~2× overconfident at B0). Calibration adds −0.0006 ALC on top of embeddings (−0.0005).
- **ECE (formative 0.151):** locally ~0.14 (pair-mean) and **not improved** by calibration (+0.003). The ECE floor for the same predictions if perfectly calibrated is 0.058 (B0 0.043, B31 0.075; `logs/ece_floor_v3.txt`), and at B0 the measured 0.20 is irreducible (all predictions ≈0.5 while pair accuracies range 0.1–0.9). ECE is a diagnostic; the lever is information (labels), not post-hoc maps.
- **labeling.py (`submission3/labeling.py`, `tests/test_v3.py`, 17 tests OK):** explicit `acquisition_function(input, *, labeled, context)` = the kit's default random policy (reveal iff sha256-uniform(context, input) < labels_remaining / items_remaining), `prediction` omitted from the signature so the evaluator makes no extra predict() calls (AcquisitionHook resolves to `stream`); plus `binarize_response`, `aggregate_responses` (only exact 0/1, NaN/graded dropped) and `construct_label` (hash, not RNG state). Tests: decisions byte-equal to the kit rule on 5,000 random contexts (and the kit source lines asserted present), deterministic, exactly 31 labels when ≥31 items and never more, nested budgets are prefixes, edge contexts. No performance claim: it reproduces the default policy; an active policy was not evaluated.
- **Packaging/checks:** `python build_zip.py submission3 dist/submission-v3.zip` (13 files, 15.05 MB: model.py, featurize.py, labeling.py, embed.py, artifacts/{ws,wi,idf,pca_mean,pca_basis}.f32, embed_table.f16, embed_vocab.txt, meta.json, calib.json). Organisers' checker (`cd paiec_baseline && ../.venv/bin/python -I check_submission_zip.py ../dist/submission-v3.zip`): OK, also `--static-only`: OK (`logs/check_v3.txt`); organiser tools tests 19 OK (`logs/organiser_tests_v3.txt`). Runtime (fresh process, extracted zip, 10 pairs/310 labels): first call 0.29 s (fit + encode), median 0.15 ms, shipped model peak RSS ≈ 43 MB (`logs/runtime_v3.txt`).
- Final prior = v2's frozen prior on all 167 benchmarks (only the test-time layer, embedding basis and calibrator are new); the shipped calibrator is fitted on the cross-fitted outer predictions of all five folds.

### 4. Caveats
- Effect is small (ALC −0.0011); the upper CI bound at S=10 is −0.0002 so the gate passes narrowly. A real-data check is the formative submission (needs Alex's go): local ALC differences of 0.001 may not show through its noise.
- Local episodes are a model of the hidden protocol (sampling of subjects/pairs/item sharing UNKNOWN; S=10 vs S=10i bracket it). Binary benchmarks only. Test-time sds other than su/k/d are v2's, tuned earlier on v2's eval folds (spread ~0.002 ALC).
- Stdlib embedding cost: ~5–10 ms per unseen item; first fit at budget 31 ≈0.3 s locally; platform per-call timeout UNKNOWN.
- The subject-group definition (model identity) is an assumption; organisers' subjects may be finer-grained (e.g. harness variants), in which case unseen-subject is easier than modelled here.
- `/tmp/m2v` (scratch venv with model2vec, used only to validate the encoder) is outside the repo; work/models/potion-base-8M is git-ignored (MIT licence per model card).

### 5. Reproduce v3
```
python dev3/fetch_model.py                                              # pinned potion-base-8M (rev bf8b0566…, sha256-checked; xet transfer disabled)
python dev3/groups.py; python dev3/build_episodes.py                          # subject groups, unseen-subject episodes (work/episodes3)
python dev3/make_embed_artifacts.py                                         # potion-base-8M -> submission3/artifacts/embed_{vocab.txt,table.f16}
python dev3/embcache.py train; python dev3/embcache.py episodes; python dev3/priors.py   # caches, 25 priors (work/art3)
for m in const05 empirical_mean v2; do python dev3/eval3.py --model $m --tag $m --S 10; done   # also S 3, 10i
dev3/tune.sh; dev3/tune2.sh; python dev3/select.py                          # inner grid + selection
dev3/stage_calib_runs.sh; python dev3/calibrate.py --tag v3raw --out work/calib_v3.json --final work/calib_final_v3.json
for S in 10 3 10i; do python dev3/eval3.py --model v3 --tag v3cal --S $S --opts '{"k":32,"su":0.15}' --calib work/calib_v3.json; done
python dev3/analyze.py --S 10 --models const05:const05 kit_empirical_mean:empirical_mean v2:v2 v3:v3cal --ref v2 --kit kit_empirical_mean
python dev3/make_submission3.py --k 32 --su 0.15 --d 1.5 --calib work/calib_final_v3.json; python build_zip.py submission3 dist/submission-v3.zip
# dev3/check_embed.py (encoder vs model2vec) needs a separate env with model2vec: uv venv /tmp/m2v && uv pip install --python /tmp/m2v/bin/python model2vec numpy; /tmp/m2v/bin/python dev3/check_embed.py
```

### 6. Next steps (hours)
1. **Alex decides on a formative upload of v3** (one slot; the only real-data check; local gain is 0.001 ALC): upload + record scores 0.5 h of agent time. Needs Alex's approval.
2. **Active acquisition in labeling.py** (uncertainty / disagreement-weighted reveal; needs a sequential simulator that predicts at acquisition time): 4–5 h incl. nested evaluation. B1–B7 dominate ALC, so this is the largest remaining lever; independent of Alex.
3. **OpenReview report (due 2026-10-30 AoE)**: methods, the unseen-subject validation, negative results (item-text prior, ECE not fixable post hoc), disclosures (potion-base-8M, LLM coding assistant), released-code README: 5–6 h; Alex handles OpenReview activation (reminder 2026-10-23).
