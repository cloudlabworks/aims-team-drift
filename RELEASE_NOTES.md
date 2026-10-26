# Local release preparation

This new repository has no inherited Git history and no configured remote. No publication, push, or submission was performed.

## Code integrity

All Python files in `submission/` and `submission3/`, and `build_zip.py`, are byte-identical to the selected local development source. Identity and isolated-checkout tests are recorded in `report/VALIDATION.json`. The original local artifact hash manifest is retained; weights themselves are excluded. Historical upload byte identity cannot be certified without upload receipts.

Development shell wrappers were relocated to their own repository root using their script path, and given `set -euo pipefail` so pipeline failures are visible. `dev/sweep2.sh` now points to the existing `dev/variants/model_k.py` rather than its obsolete submission path. No scoring algorithm, hyperparameter, or scientific result was changed.

Scientific main-text numeric-token counts and every original HTML source-citation comment are unchanged from the input report. Release-status and reconstruction wording was updated, and the selected submission ID is explicit.

## Scope exclusions

No gated measurement-db bytes or derived per-item tables are included. Training caches, subject-group maps, episode tables, per-target predictions, virtual environments, pretrained weights, trained prior arrays, projections, calibration artifacts and submission ZIPs are excluded. Benchmark-name manifests, release-eligibility metadata, code, synthetic-fixture generator and aggregate result summaries remain. No distinct competition-response training dataset needs distribution.

The source-only release requires authorised data retrieval and artifact reconstruction before prediction. Tests were run in a separate temporary checkout with original local artifacts, not by placing weights into this tree. Full retraining and hosted rescoring were not performed.
