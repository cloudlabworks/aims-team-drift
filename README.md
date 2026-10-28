# Team Drift — AIMS / NeurIPS 2026 PAIEC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23720944.svg)](https://doi.org/10.5281/zenodo.23720944)

Cloud Lab Works. Selected Codabench submission: **981740 (v2)**. The later **983242 (v3)** is included for reproducibility; its hosted primary score regressed.

**Local release candidate only: not published or pushed.** Intended public URL: https://github.com/cloudlabworks/aims-team-drift . The report uses this explicitly unverified placeholder. Publication and OpenReview submission remain separate actions.

## Contents and data boundary

- `submission/`: selected v2 predictor and featurizer, unchanged from the inspected development tree.
- `submission3/`: v3 predictor, encoder, featurizer and acquisition code.
- `dev/`, `dev3/`, `baselines/`, `tests/`: development, training, local evaluation, ablations and synthetic interface fixtures.
- `build_zip.py`: deterministic packaging used by both variants.
- `report/`: Markdown, generated LaTeX, official NeurIPS 2026 style, PDF, provenance and source/checksum manifests.
- `logs/`: aggregate scientific result summaries and sanitized historical interface/runtime checks only. No per-item result tables.
- `work/release.json`: benchmark-level public/withheld selector; not permission to redistribute gated dataset contents.

No measurement-db data, derived per-item tables, cached features, subject-group tables, model weights, fitted artifacts, virtual environments, or ZIP submissions are distributed. **Even small prior arrays are excluded as model weights.** A source-only checkout cannot predict or pass artifact-dependent tests until artifacts are rebuilt. No additional competition-specific training response dataset was used beyond measurement-db; the synthetic generator is supplied as code, not a fitted-training dataset. The pinned pretrained embedding model must be fetched separately under its own terms. See `THIRD_PARTY.md`.

Historical `work/PLAN.md`, `work/RESULTS.md`, logs and report manifests are evidence, not current submission status. Artifact checksum entries describe original local artifacts that are intentionally absent here. Source numeric results and citations are retained; no new hosted evaluation is claimed.

## 1. Environment and organiser checker

Run all commands below from this repository's root in a separate reproduction checkout. Use Python 3.12 (observed 3.12.15). The lock is an inventory of the inspected environment, not a retrospectively certified historical lock. The optional `dev3/check_embed.py` reference-encoder comparison additionally needs `model2vec`, which was not installed in this inspected environment and is not pinned by this lock. Full training is substantial CPU/RAM/disk work and was not rerun during release preparation.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
mkdir -p work/results submission/artifacts submission3/artifacts dist
git clone https://github.com/aims-foundations/paiec_baseline.git paiec_baseline
git -C paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39
```

## 2. Authorised data retrieval and v2 training

Request access and accept the dataset terms at https://huggingface.co/datasets/aims-foundations/measurement-db . Authenticate locally with `hf auth login`; do not put credentials in commands or repository files. Retrieve the complete pinned snapshot. The historical splitter permutes the complete binary cache before excluding withheld benchmarks; a partial download changes folds. Withheld sources remain excluded from final fitting and scoring. Earlier exploration used some subsequently excluded benchmarks, as disclosed in the report.

```bash
python - <<'PY'
import json
from huggingface_hub import snapshot_download
m = json.load(open('report/TRAINING_SOURCES.json'))
revs = list(m['download_metadata_revisions_for_selected_tables'])
assert len(revs) == 1
snapshot_download(repo_id='aims-foundations/measurement-db', repo_type='dataset',
                  revision=revs[0], local_dir='data/measurement-db')
PY
python dev/catalog.py
python dev/prep.py
python dev/feats.py
python dev/train_final.py 0.01 0.01 0.5
python build_zip.py
python -I -m unittest discover -s tests -p 'test_model.py' -v
(cd paiec_baseline && ../.venv/bin/python -I check_submission_zip.py ../dist/submission.zip)
```

Compare retained benchmark names with `report/TRAINING_SOURCES.json` and regenerated hashes with `report/ARTIFACT_SHA256.json`. Hash equality is an optional historical-artifact comparison, not guaranteed after retraining or proof of historical upload identity. Never commit generated data/artifacts.

## 3. Reference local validation

```bash
python -c "import sys; sys.path.insert(0,'dev'); import harness as H; H.build_all(3); H.build_all(10); H.build_all(10,indep=True)"
for S in 3 10 10i; do
  for M in const05 empirical_mean stdlib_baseline; do
    python dev/run_baselines.py "$M" "$S"
  done
  python dev/eval_model.py --S "$S" --tag final
done
for S in 3 10 10i; do
  python dev/report.py "$S" const05 empirical_mean stdlib_baseline final
done
```

Historical comparison: `logs/results_table_v2.md`. Use fresh caches. Sweep scripts preserve original argument choices and are exploratory, not clean held-out confirmation. They now locate the checkout relative to the script; their interpreter remains `.venv/bin/python`.

## 4. v3 diagonal validation, selection, calibration and packaging

Run after steps 1–3. The fetch script pins the pretrained revision and verifies source-file SHA-256 values. Its download and generated embedding table are intentionally not part of this release.

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
bash dev3/tune.sh
bash dev3/tune2.sh
python dev3/select.py
bash dev3/stage_calib_runs.sh
python dev3/calibrate.py --tag v3raw --out work/calib_v3.json --final work/calib_final_v3.json
for S in 10 3 10i; do
  python dev3/eval3.py --model v3 --tag v3cal --S "$S" --opts '{"k":32,"su":0.15}' --calib work/calib_v3.json
done
python dev3/analyze.py --S 10 --models const05:const05 kit_empirical_mean:empirical_mean v2:v2 v3_emb_only:v3emb v3:v3cal --ref v2 --kit kit_empirical_mean
python dev3/make_submission3.py --k 32 --su 0.15 --d 1.5 --calib work/calib_final_v3.json
python build_zip.py submission3 dist/submission-v3.zip
python -I -m unittest discover -s tests -p 'test_v3.py' -v
(cd paiec_baseline && ../.venv/bin/python -I check_submission_zip.py ../dist/submission-v3.zip)
```

Historical comparison: `logs/analysis_v3_main.md`, `logs/select_v3.txt`, `logs/calibrate_v3.txt`. The local validation protocol is not the hosted test; these commands cannot reproduce hidden hosted scores.

## 5. Report

```bash
brew install tectonic pandoc
python3 report/build_report.py
(cd report && tectonic report.tex)
```

The build script converts the Markdown with Pandoc and wraps it in the unmodified official 2026 NeurIPS style, final/non-anonymous mode. References and appendices start on separate pages. `report/STYLE_SOURCE.json` pins the official archive and extracted style. `RELEASE_CHECK.md` records checks, exclusions and the exact file-size inventory.

## License

Original project code: MIT, Copyright (c) 2026 Cloud Lab Works LLC. Third-party style and external dependencies retain their own terms; see `THIRD_PARTY.md`.

## Citation

If you use this project, please cite the accompanying preprint:
[Predicting AI-System Responses Without Running Them: A Hierarchical Test-Time Update for the NeurIPS 2026 Predictive AI Evaluation Competition](https://doi.org/10.5281/zenodo.23720944). DOI: [10.5281/zenodo.23720944](https://doi.org/10.5281/zenodo.23720944).
See the [paper page](https://cloudlabworks.dev/papers/predictive-ai-evaluation) for the abstract and PDF.

```bibtex
@techreport{alvord2026predictive,
  author = {Alvord, Alex},
  title = {{Predicting AI-System Responses Without Running Them: A Hierarchical Test-Time Update for the NeurIPS 2026 Predictive AI Evaluation Competition}},
  institution = {CloudLab Works},
  publisher = {Zenodo},
  year = {2026},
  month = oct,
  type = {Preprint},
  doi = {10.5281/zenodo.23720944},
  url = {https://doi.org/10.5281/zenodo.23720944}
}
```
