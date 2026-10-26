# Evidence and provenance for the draft

This record is part of the repository so numerical report claims have repository-local citations. It is not an independent hosted-score export.

## Hosted score transcription (secondary evidence)

Source: the task supplied to the report drafter and the “Next” entry in the private task record `memory/tasks/neurips-aims-aimo-2026-10.md`, read for this task. Only the following non-personal competition facts were transcribed. The private task record itself is not copied or released.

| Variant | Codabench submission ID | Recorded status | Primary field `brier` | Diagnostic `calibration_ece` |
|---|---|---|---|---|
| v2, reference | 981740 | Finished | 0.17587 | 0.15092 |
| v3, regression | 983242 | Finished | 0.18878 | 0.14418 |

Derived arithmetic from these displayed values: v3 minus v2 primary score = +0.01291; ECE = −0.00674. Lower is better in each case. These are differences between separate formative submissions, not paired-test estimates.

Repository corroboration is limited: `aims/work/PLAN.md` explicitly records the earlier hosted primary value 0.17587 and discusses uncertainty about the historical `brier` field. It predates the v3 hosted result. Primary scoring exports and per-budget feedback are UNKNOWN and were not retrieved in this draft-only task. These historical values must not be presented as newly rechecked server results or a final common-test comparison.

## Source location and chronology corrections

- The requested `aims/RESULTS.md` is absent from this checkout and its Git history. The actual tracked results document is `aims/work/RESULTS.md`; it was read in full.
- `REPORT.md` is an early local-baseline snapshot. Its statements that no AIMS training data or hosted scores were available are stale relative to later artifacts and the supplied task record.
- `aims/work/RESULTS.md` and `aims/work/PLAN.md` include historical “nothing submitted” / “ready” wording. The report explicitly supersedes those status claims, not the recorded local measurements.
- `aims/NOTES_scope.txt` was read; it is housekeeping, not scientific result evidence.
- Baseline repository revision observed locally: `82d330ddcdb16016a3ae9e048db7e588ba2c6a39`.
- Competitions source revision before drafting: `b3730a2`. Relevant preceding commits: `3488d7f` (v3 development and local validation), `bef404e` (public/withheld filter and final prior), `2f5d78c` (v2 predictor), `dda025a` (earlier baseline), `854de35` (local harness).
- The Git log's commit messages are provenance, not evidence of hosted success.

## Data and model identity

`TRAINING_SOURCES.json` was generated from `aims/work/feats/*.pkl` and `aims/work/release.json`, using the same public-release selection as the final trainer. It lists all 167 selected benchmark names and the excluded names imagenet_hard, swe_together, and visual_memory.

The Hugging Face download metadata for all 501 selected subject/item/response tables identifies measurement-db revision `54074453da7b9911be0f48e4c560be547bc84d21`. No metadata file was missing. This is local metadata evidence, not a claim that historical table bytes were independently revalidated.

The v3 pretrained model source and pinned revision are in `aims/dev3/fetch_model.py`: `minishlab/potion-base-8M`, revision `bf8b056651a2c21b8d2565580b8569da283cab23`; that script also records source-file SHA-256 expectations. The report does not claim an audit of the pretrained model's original pretraining corpus.

`ARTIFACT_SHA256.json` records current local hashes of predictor sources and artifacts for both variants. They support future reproducibility checks; absent the uploaded archive receipts, they do not independently establish byte identity with the hosted submissions.

## Current development environment (not a historical lock)

Observed Python 3.12.15. Installed versions: numpy 2.5.4; pandas 3.0.6; pyarrow 26.0.0; scipy 1.18.1; scikit-learn 1.9.1; pyyaml 6.0.3; huggingface_hub 2.2.0. No safetensors package is installed; the embedding conversion reads that file format directly with struct/json/numpy.

This inventory was read from `aims/.venv`, not inferred from historical logs. Full retraining was not run for the report draft. The development evaluator uses Python syntax requiring Python 3.12; the packaged predictors use the standard library.

Early interface development also used the synthetic Rasch-style generator in `aims/tests/synthetic_alc.py`, as described in `REPORT.md`. It is disclosed in the report, not treated as a real training or hosted score. The final prior-training code reads measurement-db caches.

## Local result sources

- `aims/logs/results_table_v2.md`: earlier benchmark-fold means and standard deviations.
- `aims/logs/analysis_v3_main.md`: diagonal-split per-budget values, cluster-bootstrap intervals, ECE diagnostics, and paired deltas.
- `aims/logs/select_v3.txt`, `aims/logs/calibrate_v3.txt`: nested selection and calibration.
- `aims/logs/check_final.txt`, `aims/logs/check_v3.txt`, `aims/logs/organiser_tests_v2.txt`, `aims/logs/organiser_tests_v3.txt`: historical interface checks, not hosted predictive evaluation.
- `aims/logs/runtime_v2.txt`, `aims/logs/runtime_v3.txt`: different benchmark runtime fixtures; not a controlled same-input speed comparison.
- `aims/logs/ablation_table.md` and `aims/work/RESULTS.md`: exploratory sweeps with the documented earlier withheld-source caveat; not promoted to clean held-out headline results.

The current organiser protocol and report requirements are quoted with their URLs in `REQUIREMENTS.md`.
