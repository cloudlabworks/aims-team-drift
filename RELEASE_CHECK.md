# Release check — 2026-10-26

## Scope and status

Published 2026-10-26 as https://github.com/cloudlabworks/aims-team-drift. Git identity: `Cloud Lab Works` <`325058966+cloudlabworks@users.noreply.github.com`> (the first commit's personal-account noreply was rewritten before the repository was made public).

Selected submission: **981740 (v2)**; v3 **983242** is included as the documented hosted regression. Intended, not-yet-published URL: https://github.com/cloudlabworks/aims-team-drift .

## Exclusions

PASS: no measurement-db data or derived per-item tables; no training/episode/result caches; no virtual environment; no model weights (including small prior arrays, PCA and fitted calibration); no model vocabulary; no submission ZIPs; no symlinks; no compiler intermediates; no file above **5,000,000 bytes**. No original home-directory paths remain. Benchmark-name/source manifests and aggregate scientific summaries are included. No additional competition-response training data differ from measurement-db. Synthetic interface-fixture code is included; generated fixture datasets are not.

## Secret and phone scan

PASS: **no credentials, private identifiers or phone-number candidates detected**. The scan is heuristic, not a guarantee about arbitrary encoded secrets. Every release content file was scanned as bytes with grep, including the PDF; the PDF's extracted text was also scanned. Internal `.git/` metadata is not release content and is excluded from the inventory and scan.

Reproduction command after staging/committing (the audit document itself contains these deliberate control strings):

```bash
git ls-files -z | xargs -0 grep -aEin 'hf_|sk-|ghp_|github_pat_|AKIA|BEGIN PRIVATE KEY|@nutanix|nutanix|alvordag|rebl_botz|zohomail'
```

Four matches outside this audit were reviewed:

- `dev3/fetch_model.py:7`: public library environment identifier `HF_HUB_DISABLE_XET`.
- `dev3/fetch_model.py:8` and `:20`: public function identifier `hf_hub_download`.
- `report/report.tex:29`: generated section label `task-and-evaluation-objective`; its substring `sk-` is not a token.

The audit document's pattern and the explanations above are intentional scan controls. No other token-prefix or private-identifier matches. Phone scan: zero candidates, using the following extended regex with case-insensitive binary-safe grep:

```text
(^|[^[:alnum:]_.])((\+?1[ .-]?)?(\([2-9][0-9]{2}\)|[2-9][0-9]{2})[ .-][2-9][0-9]{2}[ .-][0-9]{4}|\+?[1-9][0-9]{9,14})($|[^[:alnum:]_.])
```

## Validation

- All 7 v2 and 17 v3 unit tests passed; both reconstructed ZIPs passed the organiser's checker.
- These checks used an isolated temporary checkout of release code with original local artifacts. Weights and ZIPs were never placed in this release tree. No full retraining or hosted rescoring was performed.
- All Python sources compiled under Python 3.12.15. Both submission source trees and `build_zip.py` are byte-identical to the inspected development source. `report/VALIDATION.json` records commands, results and source identity.
- Every original report HTML source-citation comment is retained; scientific main-text numeric-token multisets are unchanged. Unicode minus signs were explicitly mapped to mathematical minus glyphs and checked in PDF text.
- Development shell wrappers use checkout-relative paths and fail on pipeline errors; an obsolete exploratory variant path was repaired. Full rerun order and environment inventory: `README.md`, `requirements.txt`, `requirements.lock`. The lock is current inventory, not a historical guarantee.

## NeurIPS PDF

Official 2026 archive, linked from the 2026 Call for Papers:

- URL: https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
- Archive SHA-256: `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`
- Extracted member: `neurips_2026.sty`
- Style SHA-256: `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`

The obsolete StyleFiles page and guessed direct style path returned 404, but the actual official **2026** archive was available; no older-year fallback was needed. The style is unmodified; final/non-anonymous workshop mode is used.

`report/report.pdf`: **6 main-text pages including Acknowledgements / 10 total**. References: page 7. Appendices: pages 8–10. Instruments: Poppler `pdfinfo` and pypdf page-text boundary inspection. Title and result-page renders were inspected; the hosted table is kept together. Only underfull text-box warnings remain; no missing glyphs or overfull boxes. No facts or scientific numbers were trimmed.

PDF SHA-256: `a7938438110d811148dc323888c8c6e08e9a06c1637af0b0e0b06fac08343414`.

Tools installed with Homebrew: Tectonic 0.17.0 and Pandoc 3.12.1. Initial proxy-certificate failures were resolved for public downloads with command-local environment cleanup. The paired-node execution route returned a state-database contention error; execution used the working local macOS Gateway host on the same filesystem.

## Remaining external actions / limitations

Publication and OpenReview submission were deliberately not performed. The release URL is an explicit placeholder until publication. Author review, historical hosted receipts, a full early-development inventory audit, and proof of bit-identical full retraining remain outstanding as already disclosed in the report. Source-only prediction requires authorised data access and reconstruction of excluded artifacts. The optional model2vec reference-encoder checker needs its separately documented optional dependency.

## Every release file and exact size

Bytes, including this audit file; excludes internal Git metadata. This is the complete release-content inventory, not only selected examples.

| File | Bytes |
|---|---:|
| `.gitignore` | 226 |
| `LICENSE` | 1076 |
| `README.md` | 7519 |
| `RELEASE_CHECK.md` | 8501 |
| `RELEASE_NOTES.md` | 1909 |
| `THIRD_PARTY.md` | 1212 |
| `baselines/stdlib_model.py` | 2971 |
| `build_zip.py` | 1050 |
| `dev/ablation_table.py` | 1772 |
| `dev/catalog.py` | 1163 |
| `dev/diag_zi.py` | 1377 |
| `dev/eval_model.py` | 3310 |
| `dev/eval_prior.py` | 1574 |
| `dev/feats.py` | 1817 |
| `dev/final_eval.sh` | 230 |
| `dev/folds.py` | 461 |
| `dev/harness.py` | 4636 |
| `dev/prep.py` | 2527 |
| `dev/report.py` | 1093 |
| `dev/run_baselines.py` | 1163 |
| `dev/runtime_test.py` | 1389 |
| `dev/sweep1.sh` | 475 |
| `dev/sweep2.sh` | 497 |
| `dev/sweep3.sh` | 479 |
| `dev/sweep4.sh` | 361 |
| `dev/train_final.py` | 757 |
| `dev/train_prior.py` | 3455 |
| `dev/variants/model_ev.py` | 7732 |
| `dev/variants/model_k.py` | 7869 |
| `dev3/analyze.py` | 6376 |
| `dev3/build_episodes.py` | 508 |
| `dev3/calibrate.py` | 5125 |
| `dev3/check_embed.py` | 1377 |
| `dev3/ece_floor.py` | 1096 |
| `dev3/embcache.py` | 2890 |
| `dev3/eval3.py` | 4670 |
| `dev3/fetch_model.py` | 1178 |
| `dev3/groups.py` | 3462 |
| `dev3/harness3.py` | 6668 |
| `dev3/make_embed_artifacts.py` | 754 |
| `dev3/make_submission3.py` | 1621 |
| `dev3/priors.py` | 2242 |
| `dev3/runtime_test3.py` | 2019 |
| `dev3/select.py` | 1624 |
| `dev3/stage_calib_runs.sh` | 719 |
| `dev3/tune.sh` | 675 |
| `dev3/tune2.sh` | 733 |
| `logs/ablation_table.md` | 2682 |
| `logs/analysis_v3_main.md` | 38457 |
| `logs/calibrate_v3.txt` | 704 |
| `logs/check_final.txt` | 283 |
| `logs/check_v3.txt` | 300 |
| `logs/organiser_tests_v2.txt` | 118 |
| `logs/organiser_tests_v3.txt` | 98 |
| `logs/results_table_v2.md` | 2119 |
| `logs/runtime_v2.txt` | 574 |
| `logs/runtime_v3.txt` | 825 |
| `logs/select_v3.txt` | 1675 |
| `report/ARTIFACT_SHA256.json` | 1941 |
| `report/CHECKLIST.md` | 6744 |
| `report/EVIDENCE.md` | 5595 |
| `report/PDF_CHECK.json` | 697 |
| `report/REQUIREMENTS.md` | 9470 |
| `report/STYLE_SOURCE.json` | 537 |
| `report/TRAINING_SOURCES.json` | 3866 |
| `report/VALIDATION.json` | 4683 |
| `report/build_report.py` | 2841 |
| `report/neurips_2026.sty` | 13704 |
| `report/report.md` | 32003 |
| `report/report.pdf` | 66877 |
| `report/report.tex` | 31440 |
| `requirements.lock` | 515 |
| `requirements.txt` | 63 |
| `submission/featurize.py` | 2645 |
| `submission/model.py` | 7116 |
| `submission3/embed.py` | 4717 |
| `submission3/featurize.py` | 2645 |
| `submission3/labeling.py` | 2748 |
| `submission3/model.py` | 11681 |
| `tests/synthetic_alc.py` | 3219 |
| `tests/test_model.py` | 3219 |
| `tests/test_v3.py` | 9363 |
| `work/PLAN.md` | 21164 |
| `work/RESULTS.md` | 12376 |
| `work/release.json` | 7045 |

**Total: 85 files, 429,087 bytes.**
