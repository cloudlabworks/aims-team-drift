# Team Drift: Hierarchical Test-Time Adaptation for Predictive AI Evaluation

**Alex Alvord, Cloud Lab Works**

**Team:** Team Drift
**NeurIPS 2026 Predictive AI Evaluation Competition (AIMS / PAIEC)**

**Selected Codabench submission:** 981740 (v2)
**Status:** Technical-report draft; not submitted.
<!-- Sources: aims/report/REQUIREMENTS.md (competition and venue); aims/report/EVIDENCE.md (submission identity and supplied task record). -->

> Release candidate: https://github.com/cloudlabworks/aims-team-drift (intended public URL; not yet published or verified). This report is typeset with the official NeurIPS template. The source-only release excludes gated measurement-db data, derived per-item tables, all model weights and fitted artifacts. Rebuild artifacts before the verification commands in Appendix A; the release README gives the complete ordered commands. No additional response training dataset differs from measurement-db. No publication or submission is claimed.

## Abstract

We study probability prediction for unobserved AI-system responses under benchmark-level cold start and a limited budget of revealed labels. Our reference method combines a regularized cross-benchmark logistic prior with a hierarchical logistic update fitted to the permitted test-time evidence. It shares information through benchmark shifts, subject–benchmark offsets, and item effects across subjects, while requiring only standard-library computation at inference. A later variant adds static item embeddings, deterministic random acquisition, and budget-specific temperature calibration. Local unseen-subject validation favored that extension: Brier area under the learning curve (ALC) decreased from 0.2060 to 0.2049. Hosted formative feedback reversed the ranking: the recorded primary `brier` value increased from 0.17587 to 0.18878, despite a lower calibration-error diagnostic. We therefore retain v2 as the reference and report v3 as a hosted regression. The result illustrates a limitation of local model selection: an internally consistent validation improvement does not establish improvement under a separately sampled hosted evaluation.
<!-- Sources: aims/logs/analysis_v3_main.md, first S=10 table; aims/report/EVIDENCE.md, hosted-score transcription. -->

## 1. Task and evaluation objective

The competition asks a submitted predictor to estimate the probability that a subject answers an item correctly, without running that subject on the item. A subject includes its model configuration and evaluation harness; an item includes the task and supplied attributes. The evaluation is a benchmark-level cold start. This setting differs from simply completing a response matrix: the predictor must work on benchmark items unavailable during training, and subject coverage may also change.

The current organiser page defines the primary metric as pair-equal Brier ALC. Brier error is measured on the same evaluation responses at label budgets 0, 1, 3, 7, 15, and 31. For a subject–benchmark pair, with budget-specific error denoted by $B_n$,

$$
\operatorname{ALC}=0.1B_0+0.2B_1+0.2B_3+0.2B_7+0.2B_{15}+0.1B_{31}.
$$

Pair-level ALC values are then averaged equally. These weights correspond to trapezoidal integration on the logarithmic budget axis. Expected calibration error (ECE) is diagnostic, not the primary selection criterion. The page describes ECE using 10 equal-width probability bins and applies the ALC weights across budgets.
<!-- Sources: aims/report/REQUIREMENTS.md, Evaluation facts; aims/dev/harness.py, BUDGETS and W. Mathematical constants are metric definitions, not measurements. -->

Acquisition and evaluation items are separated within eligible pairs; revealed acquisition outcomes are not scored. Our predictor uses only the supplied evidence, never requests subject inference, and performs no runtime network access. The historical hosted field is reported below under its recorded name, `brier`. Although the current page specifies Brier ALC, the archival scorer definition associated with those earlier submissions is unavailable here; we do not silently relabel a historical field using a later page snapshot.
<!-- Sources: aims/submission/model.py; aims/submission3/model.py; aims/work/PLAN.md; aims/report/REQUIREMENTS.md; aims/report/EVIDENCE.md. -->

## 2. Data, training sources, and preprocessing

The competition-specific response data come from the organiser's [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db). The final prior uses 167 public-release binary benchmarks. The source manifest accompanying this report enumerates every selected benchmark. Local download metadata identifies revision `54074453da7b9911be0f48e4c560be547bc84d21`; this is a recovered cache revision, not an independently certified historical download receipt. Benchmarks marked non-public in the release map are excluded from the final prior, including imagenet_hard, swe_together, and visual_memory.
<!-- Sources: aims/submission/artifacts/meta.json; aims/dev/train_final.py; aims/dev/folds.py; aims/report/TRAINING_SOURCES.json; aims/report/EVIDENCE.md. -->

Preprocessing discards missing responses and retains benchmarks whose observed response values are binary. Repeated subject–item trials are aggregated into a trial count and a success count, preserving their contribution to the likelihood and local Brier evaluation. Subject attributes include normalized model name, provider, release/access information, harness/version, reasoning effort, and additional supplied features. Item inputs include content, item features, and interactors. Truncation and hashing are deterministic; the benchmark-ID field is not an offline predictive feature. The feature cache limits training to 10,000 sampled items per benchmark to bound computation.
<!-- Sources: aims/dev/catalog.py; aims/dev/prep.py; aims/dev/feats.py; aims/submission/featurize.py. -->

The reference model does not use a pretrained language model. The extension uses the pretrained static embedding model [minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M), pinned to revision `bf8b056651a2c21b8d2565580b8569da283cab23`. Its vocabulary and embedding table are bundled for inference. Competition-specific projection fitting, prior training, and calibration use the local competition-data pipeline; no additional response dataset is introduced by this extension. We disclose the pretrained model separately rather than claiming knowledge of, or auditing, its original pretraining data.
<!-- Sources: aims/dev3/fetch_model.py; aims/dev3/make_embed_artifacts.py; aims/dev3/priors.py; aims/dev3/calibrate.py. -->

A qualification applies to development history: earlier exploratory sweeps included the subsequently excluded withheld-release benchmarks. The retained final prior applies the public-release filter, but this does not retroactively remove their role in exploration. Those sweeps are not treated as clean held-out evidence in this report. UNKNOWN: complete inventory-level compliance audit of that earlier development data. Graded-response benchmarks were not assigned a new binarization rule for training.
<!-- Sources: aims/work/RESULTS.md, Ablations / sweeps and Limitations; aims/dev/folds.py; aims/dev/prep.py. -->

Early development also used a synthetic Rasch-style response simulator for interface and baseline checks. Its assumptions resemble the additive-logit method, so those checks are not evidence of real benchmark generalization. Synthetic scores are excluded from the result tables, and the final prior-training pipeline reads measurement-db caches rather than simulated responses. The broader development-source audit should include these fixtures.
<!-- Sources: REPORT.md, AIMS baseline caveats; aims/tests/synthetic_alc.py; aims/dev/train_final.py; aims/dev/train_prior.py. -->

## 3. Reference approach: offline prior and hierarchical adaptation

### 3.1 Offline prior

Let $z_s(s)$ be a learned subject-attribute effect and $z_i(i)$ a learned item-text effect. The prior is

$$
\eta_{si}^{\mathrm{prior}}=b+z_s(s)+z_i(i),\qquad p_{si}=\sigma(\eta_{si}^{\mathrm{prior}}).
$$

Subject attributes and selected interactions are hashed into binary features. Item content is represented by hashed unigrams, bigrams, length indicators, and supplied feature tokens. Sublinear term frequency, training-derived inverse document frequency, and vector normalization produce the item representation. Logistic ridge regression fits the intercept and feature weights using aggregated binomial response counts.
<!-- Sources: aims/submission/featurize.py; aims/dev/train_prior.py. -->

Each benchmark's likelihood is weighted so its total contribution scales with the square root of its retained trial count. The shipped subject and item ridge coefficients are both 0.01, with benchmark-weighting exponent 0.5. This compromises between pooling all responses indiscriminately and giving each benchmark identical training weight. It is not identical to the evaluation's equal-pair aggregation, a deliberate but potentially consequential difference.
<!-- Sources: aims/dev/train_prior.py, fit_prior; aims/submission/artifacts/meta.json. -->

### 3.2 Adaptation from permitted labels

For a target in benchmark $q$, the adapted logit is

$$
\eta_{si}=b+(1+g_q)z_s(s)+z_i(i)+\beta_q+\delta_{sq}+d_{iq}.
$$

The benchmark shift $\beta_q$ adjusts overall difficulty; $g_q$ rescales the subject effect; $\delta_{sq}$ is a subject–benchmark correction; and $d_{iq}$ shares item evidence across subjects. The update filters the supplied labels to the target benchmark. Thus, when the interface supplies multiple subjects from that benchmark, evidence can be pooled across them without accessing labels from other benchmarks.
<!-- Source: aims/submission/model.py, _evidence, _fit, predict_logit. The additive constant in the slope is part of the model definition. -->

The corrections are maximum-a-posteriori estimates under independent zero-centered Gaussian priors. Their standard deviations are 1.5 for the benchmark shift, 0.35 for the subject slope, 0.8 for the subject–benchmark offset, and 1.5 for the item offset. The implementation performs 12 Newton-style coordinate sweeps with bounded steps, not full posterior integration. Predictions are clipped to [0.01, 0.99]. With no labels, the frozen offline prior is returned. Fitted updates are cached by evidence set; caching changes cost, not which evidence is available.
<!-- Source: aims/submission/model.py, PRIOR_SD, SWEEPS, CLIP, Predictor. -->

The reference package has no acquisition function and relies on the platform's random label selection. It consists of Python source and fitted numeric arrays and uses the standard library at scoring time. This keeps the deployed method inspectable and avoids dependence on a remote inference service. Its usefulness nevertheless depends on the structure of the revealed evidence: item effects cannot transfer across subjects if they do not share relevant items.
<!-- Sources: aims/build_zip.py; aims/submission/model.py; aims/report/ARTIFACT_SHA256.json; aims/work/RESULTS.md. -->

## 4. Extension: embeddings, acquisition, and calibration

The later variant retains the reference offline prior but adds a benchmark-specific linear effect on whitened principal components of static item embeddings. The selected configuration uses 32 components and a Gaussian coefficient prior with standard deviation 0.15. At test time, a regularized Newton update fits the embedding coefficients jointly with the hierarchical corrections. This allows evidence about an acquired item to influence an unseen item through its representation, even without exact item overlap.
<!-- Sources: aims/submission3/artifacts/meta.json; aims/submission3/model.py; aims/dev3/priors.py; aims/logs/select_v3.txt. -->

The explicit acquisition function reproduces the baseline kit's deterministic random-mode rule: a hash-derived uniform value is compared with labels remaining divided by candidate items remaining. It does not use prediction uncertainty, so this extension is not a claim of improved active acquisition. The local repeated-response label sampler is also deterministic in its supplied key. The organiser still controls the actual candidate stream and hidden outcomes.
<!-- Sources: aims/submission3/labeling.py; aims/dev3/harness3.py; aims/tests/test_v3.py. -->

Budget-specific calibration maps are trained on cross-fitted local predictions. Candidate families are the identity map, temperature scaling, affine-logit calibration, and isotonic calibration. Fitting the parametric maps minimizes a regularized, pair-weighted likelihood; family selection evaluates the Brier-ALC proxy. Nested selection chose temperature scaling in every outer fold. The shipped map has a zero intercept and a separately learned logit multiplier for each budget bucket; the final map is fitted using cross-fitted outer predictions.
<!-- Sources: aims/dev3/calibrate.py; aims/logs/calibrate_v3.txt; aims/submission3/artifacts/calib.json. -->

Importantly, calibration is not guaranteed to reduce Brier error after distribution shift. The local selection criterion and the local composition of subject–benchmark pairs determine the shrinkage learned by these maps. Their benefit on local folds does not imply the same benefit on the hosted formative sample. The hosted result is therefore reported as a regression rather than rationalized as a successful calibration improvement.
<!-- Sources: aims/logs/analysis_v3_main.md; aims/report/EVIDENCE.md. -->

## 5. Experimental setup

The original local protocol holds out benchmark folds while allowing subjects to recur across training and validation benchmarks. It uses five folds, nested label budgets, and equal-pair Brier aggregation within each fold. Its headline summary averages the fold means. With up to 10 sampled subjects sharing an item pool, the reference method obtained local ALC 0.1953 ± 0.0071; the unchanged kit empirical-mean predictor obtained 0.2475 ± 0.0115. These dispersions are standard deviations across folds, not confidence intervals. Test-time hyperparameters were explored on these evaluation folds, making this an optimistic development summary.
<!-- Sources: aims/logs/results_table_v2.md, S=10; aims/dev/eval_model.py; aims/work/RESULTS.md, validation design and caveats. -->

For the extension, a diagonal benchmark-and-subject holdout was constructed. Subject groups are primarily the case-folded normalized model name, joining harness and reasoning variants of the same named model. Fold-specific priors exclude both the held-out benchmark fold and the held-out subject-group fold. Inner-fold predictions support extension hyperparameter and calibrator selection. This is a stricter identity holdout, but the grouping is a local assumption rather than the organiser's definition of a subject.
<!-- Sources: aims/dev3/groups.py; aims/dev3/harness3.py; aims/dev3/priors.py; aims/dev3/select.py; aims/dev3/calibrate.py. -->

The diagonal episode builder uses five deterministic draws per benchmark and setting. The main diagonal analysis includes 1,907 pairs, 113 benchmarks, and 474 subject groups, with a maximum of 10 sampled subjects per local episode. Confidence intervals use 4,000 bootstrap resamples of benchmark clusters and the 95% percentile interval. Paired deltas compare models on the same local episodes. Additional subject-cluster bootstraps and alternative subject/item-overlap settings are preserved in the detailed analysis log. These intervals quantify local resampling variation; they do not cover hosted-sampling uncertainty or all model-selection effects.
<!-- Sources: aims/dev3/build_episodes.py; aims/logs/analysis_v3_main.md, first S=10 section; aims/dev3/analyze.py. -->

## 6. Results and the hosted regression

### 6.1 Local unseen-subject results

The table reports the diagonal-split analysis, not the earlier benchmark-only protocol. “Reference” here means the reference algorithm evaluated with fold-trained priors, not its all-data deployment artifact.

| Method | Local Brier ALC | Benchmark-bootstrap interval | Local pair-mean ECE |
|---|---:|---|---:|
| Constant probability | 0.2500 | [0.2500, 0.2500] | 0.2057 |
| Kit empirical mean | 0.2540 | [0.2392, 0.2674] | 0.1802 |
| Reference v2 | 0.2060 | [0.1950, 0.2163] | 0.1405 |
| Embedding extension without calibration | 0.2055 | [0.1946, 0.2160] | 0.1440 |
| Full v3 extension | 0.2049 | [0.1946, 0.2146] | 0.1435 |

<!-- Source for every table value: aims/logs/analysis_v3_main.md, first S=10 benchmark-cluster table; intervals are 95% as defined in Section 5. Constant probability is 0.5 in aims/dev3/eval3.py. -->

The paired local ALC change for the full extension relative to the reference is −0.0011, with interval [−0.0019, −0.0002]. This is a small local gain. Local pair-mean ECE instead increases by 0.0030. The analysis also reports response-pooled ECE, which weights the response distribution differently and must not be substituted for the pair-mean diagnostic. The earlier local ALC summary and this table likewise differ in both splitting and aggregation; they are not a chronological improvement curve.
<!-- Sources: aims/logs/analysis_v3_main.md, first paired-delta table; aims/dev/eval_model.py; aims/dev3/analyze.py. -->

### 6.2 Recorded hosted formative feedback

| Variant | Submission ID | Primary field `brier` | Diagnostic `calibration_ece` | Interpretation |
|---|---:|---:|---:|---|
| v2 | 981740 | 0.17587 | 0.15092 | Reference |
| v3 | 983242 | 0.18878 | 0.14418 | Worse primary score |

<!-- Source for every value: aims/report/EVIDENCE.md, explicitly labelled secondary transcription of the supplied task record; aims/work/PLAN.md independently corroborates the earlier 0.17587 only. -->

The observed primary-score difference is +0.01291, while ECE changes by −0.00674. Thus the local-versus-hosted ranking reversal is plain: local validation favored the extension, but the recorded hosted primary metric worsened. Lower hosted ECE does not offset that primary-metric regression. We retain submission 981740 as the report's reference, without claiming this draft selects or submits anything through the venue.
<!-- Source: aims/report/EVIDENCE.md, arithmetic from displayed hosted scores. -->

These hosted numbers are transcribed from the supplied task record, not a fresh scoring export. The current rules also state that formative submissions receive newly sampled hidden subsets. Consequently, the difference is an observed cross-submission regression, not a paired causal estimate of the extension's effect. A common hidden evaluation, or suitable repeated runs, would be needed to separate methodological degradation from sample variation. No final-test performance or ranking is claimed.
<!-- Sources: aims/report/EVIDENCE.md; aims/report/REQUIREMENTS.md, formative sampling and common summative subset quotations. -->

## 7. Implementation and reproducibility

The implementation separates offline training from standard-library prediction. The reference package contains `model.py`, `featurize.py`, and the prior arrays. The extension additionally includes a bundled static encoder, projection arrays, calibration parameters, and `labeling.py`. Development uses NumPy, SciPy, pandas, PyArrow, scikit-learn, and supporting data tools; these are not inference-time package requirements. Appendix A gives a staged reproduction path, and accompanying manifests identify training sources and current artifact checksums.
<!-- Sources: aims/build_zip.py; aims/dev/train_prior.py; aims/dev3/make_submission3.py; aims/report/ARTIFACT_SHA256.json; aims/report/TRAINING_SOURCES.json. -->

Historical organiser-checker logs report valid ZIP packages for both variants. Those checks establish interface/package validity, not predictive quality or exact equivalence with the hosted environment. The runtime logs use different benchmark fixtures, so they do not support a controlled speed ratio. Full retraining and hosted rescoring were not performed to prepare this draft. UNKNOWN: original historical environment lock and bit-identical end-to-end replay. Intended source release: https://github.com/cloudlabworks/aims-team-drift (placeholder; not yet public). The release README supplies ordered rebuild instructions; fitted artifacts and model weights are intentionally excluded from this source-only release. Public availability remains a prerequisite for the organiser's release-link requirement.
<!-- Sources: aims/logs/check_final.txt; aims/logs/check_v3.txt; aims/logs/runtime_v2.txt; aims/logs/runtime_v3.txt; aims/report/EVIDENCE.md; aims/report/REQUIREMENTS.md. -->

## 8. Limitations and conclusion

The principal limitation is validation fidelity. The local harness assumes a particular subject grouping, item-sharing pattern, episode construction, and evidence union. The hosted evaluator selects subjects and resamples formative data; its historical aggregate fields are not fully archived here. These differences are plausible contributors to the ranking reversal, but none has been isolated experimentally. We do not attribute the regression specifically to embeddings or calibration without a controlled hosted ablation.

There are further limitations. Exact-item sharing may account for gains that disappear when subjects see disjoint items. Hashed text features offer limited semantic transfer, and closely related benchmark families can cross local folds. The diagonal holdout reduces subject-identity overlap but does not guarantee independence among benchmark families. Early hyperparameter exploration and the withheld-source caveat limit the strength of clean validation claims. Bootstrap intervals do not remedy these design issues. Binary-only training may also transfer poorly to benchmarks whose original responses require other scoring conventions.
<!-- Sources: aims/work/RESULTS.md, caveats; aims/work/PLAN.md, v3 caveats; aims/dev3/groups.py; aims/dev3/harness3.py. -->

The supported conclusion is modest: hierarchical adaptation is a practical, inspectable reference method, while the tested extension offers a small local gain that did not carry over to recorded hosted feedback. The reference remains v2. Future work should first reconcile historical scoring definitions and reproduce the hosted sampling assumptions, then evaluate changes on comparable evidence. The available record does not support a claim of a winning method, a new state of the art, or a resolved explanation of the regression.
<!-- Sources: aims/logs/analysis_v3_main.md; aims/report/EVIDENCE.md; aims/report/REQUIREMENTS.md. -->

## Acknowledgements and AI-assistance disclosure

LLM-based coding assistants were used during development, and an AI assistant helped inspect the repository and draft this report. This assistance is disclosed under the competition's request to disclose coding-assistant use. Alex Alvord is the author; AI systems are not listed as authors. We acknowledge the organisers' measurement-db resource and PAIEC baseline implementation, and the pretrained static embedding model used in the extension. This disclosure does not assert that human review of this draft has already occurred.
<!-- Sources: supplied report-drafting task and agent-developed repository history, recorded in aims/report/EVIDENCE.md; organiser policy quoted verbatim in aims/report/REQUIREMENTS.md, R8. -->

## References

- AIMS organisers. Competition rules, predictor interface, and evaluation protocol. https://aims.stanford.edu/competition .
- AIMS organisers. Measurement database. https://huggingface.co/datasets/aims-foundations/measurement-db .
- AIMS organisers. PAIEC reference predictors and acquisition functions. https://github.com/aims-foundations/paiec_baseline .
- Pretrained static embedding model: https://huggingface.co/minishlab/potion-base-8M .
- Report venue: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC .

## Appendix A. Reproduction guide

**Working directory:** the source-release repository root in a separate checkout. The following existing-artifact checks require rebuilding the excluded weights first; use the ordered steps in the release README. Create `submission3/artifacts` before encoder conversion. Commands below are reproduction instructions, not evidence that a new training run was executed for this draft. Preserve original artifacts and logs when reproducing. An already reconstructed reference package can be checked without downloading training data again; the source-only release contains no such package. The organiser baseline repository is an external checkout rather than a tracked part of this repository; restore it if absent using the revision recorded in `report/EVIDENCE.md`:

```bash
git clone https://github.com/aims-foundations/paiec_baseline.git paiec_baseline
git -C paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39
```
<!-- Source: aims/report/EVIDENCE.md, observed local baseline repository revision. -->

### A.1 Verify the reference package

Use Python 3.12 for development scripts; the inspected development environment is recorded in `report/EVIDENCE.md`. The evaluator source uses syntax requiring that Python version. A minimal existing-artifact path is:

```bash
python3.12 build_zip.py
cd paiec_baseline
python3.12 -I check_submission_zip.py ../dist/submission.zip
cd ..
python3.12 -I -m unittest discover -s tests -p 'test_model.py'
```
<!-- Sources: aims/build_zip.py; aims/paiec_baseline/check_submission_zip.py; aims/tests/test_model.py; aims/dev/eval_model.py; aims/report/EVIDENCE.md. -->

Compare source and array checksums with `report/ARTIFACT_SHA256.json`. This comparison identifies the local report artifacts, not the historical uploaded archive. The large extension embedding table is not tracked in Git. A fresh checkout must regenerate it with the pinned fetch/conversion steps in Appendix A.3, or obtain the matching artifact through the eventual release. To rebuild the extension ZIP from its existing complete artifacts:

```bash
python3.12 build_zip.py submission3 dist/submission-v3.zip
cd paiec_baseline
python3.12 -I check_submission_zip.py ../dist/submission-v3.zip
cd ..
python3.12 -I -m unittest discover -s tests -p 'test_v3.py'
```
<!-- Sources: aims/build_zip.py; aims/logs/check_v3.txt; aims/tests/test_v3.py. -->

### A.2 Rebuild data caches, prior, and reference validation

Obtain authorised access to measurement-db through its dataset page. Download the revision recorded in `report/TRAINING_SOURCES.json` into `data/measurement-db`; do not silently substitute the latest dataset. Retain the tracked `work/release.json` filter and compare the resulting selected benchmark names with that manifest. Retrieve the complete pinned snapshot for a historical replay: the fold splitter permutes the full binary-feature cache before excluding non-public benchmarks, so downloading only the retained benchmarks changes fold assignments. Cache construction therefore scans excluded benchmarks too; they must remain excluded from final fitting and scoring. Install the development dependencies listed in `report/EVIDENCE.md` in a Python environment. An original historical dependency lock is unavailable. One explicit setup and data-retrieval path is:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python - <<'PY'
import json
from huggingface_hub import snapshot_download
m = json.load(open("report/TRAINING_SOURCES.json"))
revs = list(m["download_metadata_revisions_for_selected_tables"])
assert len(revs) == 1, "Resolve mixed dataset revisions before retraining"
snapshot_download(
    repo_id="aims-foundations/measurement-db", repo_type="dataset",
    revision=revs[0], local_dir="data/measurement-db",
)
PY
mkdir -p work/results submission/artifacts
```
<!-- Sources: aims/report/TRAINING_SOURCES.json; aims/dev/catalog.py; aims/dev/prep.py; aims/dev/train_prior.py; aims/dev3/calibrate.py. This is a new reproduction command, not a claimed historical run. The assertion requires a single pinned revision. -->

In a clean reproduction checkout, ensure `work/`, `work/results/`, and `submission/artifacts/` exist. Run:

```bash
python dev/catalog.py
python dev/prep.py
python dev/feats.py
python -c "import sys; sys.path.insert(0,'dev'); import harness as H; H.build_all(3); H.build_all(10); H.build_all(10,indep=True)"
for S in 3 10 10i; do
  for M in const05 empirical_mean stdlib_baseline; do
    python dev/run_baselines.py "$M" "$S"
  done
  python dev/eval_model.py --S "$S" --tag final
done
python dev/train_final.py 0.01 0.01 0.5
python build_zip.py
```
<!-- Sources: aims/work/RESULTS.md, Reproduce; aims/dev/run_baselines.py; aims/dev/eval_model.py; aims/dev/train_final.py; aims/submission/artifacts/meta.json. -->

Use fresh caches: fold-artifact filenames do not encode every data-provenance decision. In particular, old artifacts from exploratory withheld-source runs must not be reused for a claimed clean replay. Compare reference fold summaries with `logs/results_table_v2.md`, taking account of the documented fold aggregation.

### A.3 Reproduce the extension analysis

After the reference data preparation, fetch the pinned pretrained model with `dev3/fetch_model.py`, which verifies source-file hashes. Create subject groups and diagonal episodes, build the static encoder, and construct embedding caches and fold-specific priors:

```bash
python dev3/fetch_model.py
python dev3/groups.py
python dev3/build_episodes.py
python dev3/make_embed_artifacts.py
python dev3/embcache.py train
python dev3/embcache.py episodes
python dev3/priors.py
for S in 10 3 10i; do
  for M in const05 empirical_mean v2; do
    python dev3/eval3.py --model "$M" --tag "$M" --S "$S"
  done
done
```
<!-- Sources: aims/work/PLAN.md, Reproduce v3; aims/dev3/fetch_model.py; aims/dev3/build_episodes.py; aims/dev3/eval3.py. -->

The full selection path is `dev3/tune.sh`, `dev3/tune2.sh`, then `dev3/select.py`. The release shell wrappers locate the checkout relative to their script and use `.venv/bin/python`; their command arguments are preserved. The selected configuration and calibration path are:

```bash
bash dev3/stage_calib_runs.sh
python dev3/calibrate.py --tag v3raw --out work/calib_v3.json --final work/calib_final_v3.json
for S in 10 3 10i; do
  python dev3/eval3.py --model v3 --tag v3cal --S "$S" --opts '{"k":32,"su":0.15}' --calib work/calib_v3.json
done
python dev3/analyze.py --S 10 --models const05:const05 kit_empirical_mean:empirical_mean v2:v2 v3_emb_only:v3emb v3:v3cal --ref v2 --kit kit_empirical_mean
python dev3/make_submission3.py --k 32 --su 0.15 --d 1.5 --calib work/calib_final_v3.json
python build_zip.py submission3 dist/submission-v3.zip
```
<!-- Sources: aims/dev3/stage_calib_runs.sh; aims/dev3/select.py; aims/dev3/calibrate.py; aims/dev3/analyze.py; aims/dev3/make_submission3.py; aims/work/PLAN.md. -->

Compare the analysis with `logs/analysis_v3_main.md`. Neither these commands nor the ZIP checker reproduces hosted scores. UNKNOWN: primary historical scoring receipts. Intended public release location: https://github.com/cloudlabworks/aims-team-drift (not yet published).

## Appendix B. Evidence boundaries

Every numerical result and hyperparameter in this draft has an adjacent source comment. Hosted values are explicitly secondary evidence in `report/EVIDENCE.md`; local experimental values remain anchored in their original logs. The requested results document was found at `work/RESULTS.md`, not the requested top-level location. Earlier “nothing submitted” and “ready” status text in repository notes is not repeated as current fact.

The local source-release preparation and final template rendering are recorded in `RELEASE_CHECK.md`; publication and submission-form details remain open in `report/CHECKLIST.md`. No submission, upload, public release, or organiser contact is part of this draft.
