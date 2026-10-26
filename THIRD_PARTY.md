# Third-party material

- `report/neurips_2026.sty` is the unmodified official NeurIPS 2026 style. Its author notices remain intact. Source and hashes: `report/STYLE_SOURCE.json`. The project MIT notice does not override third-party terms.
- PAIEC organiser baseline and checker are external, not vendored: https://github.com/aims-foundations/paiec_baseline at revision `82d330ddcdb16016a3ae9e048db7e588ba2c6a39`. Follow that repository's licence. The acquisition rule is acknowledged in `submission3/labeling.py` and `tests/test_v3.py`.
- The v3 static model is `minishlab/potion-base-8M` (MIT), revision `bf8b056651a2c21b8d2565580b8569da283cab23`: https://huggingface.co/minishlab/potion-base-8M . No vocabulary or weight files are redistributed. Fetch/checksum script: `dev3/fetch_model.py`.
- measurement-db is gated: https://huggingface.co/datasets/aims-foundations/measurement-db . Authorised users obtain it separately under its terms. No dataset bytes or derived per-item tables are included. `work/release.json` describes original benchmark-release eligibility, not redistribution permission for measurement-db.
- Python dependencies, Pandoc and Tectonic retain their own licences and are not vendored.
