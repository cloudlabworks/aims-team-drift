# Draft readiness checklist

This checklist maps the quoted requirements in `REQUIREMENTS.md` to the draft. **Draft complete does not mean submission-ready.** No upload, submission, public release, or organiser contact is authorised or represented by this checklist.

| Requirement | Where addressed | Status / remaining work |
|---|---|---|
| R1: deadline | REQUIREMENTS.md, R1 | Recorded: 30 October 2026, 23:59 AoE. Submission itself is outside this draft task. |
| R2: selected Codabench entry | report.md title block and §6.2 | Reference is v2 / 981740. OpenReview form not filled. |
| R3: NeurIPS template | REQUIREMENTS.md, Interpretation | PREPARED: official NeurIPS 2026 style, final/non-anonymous workshop mode; `report.tex` and `report.pdf`. Exact source and hashes: `STYLE_SOURCE.json`. |
| R3: approximately 4–8 main-text pages; references and appendices unlimited | Length audit below | CHECKED: 6 main-text pages including Acknowledgements; 10 total. References page 7; appendices pages 8–10. See `PDF_CHECK.json`. |
| R4: anonymity not required | report.md title block | Alex Alvord, Cloud Lab Works; Team Drift. No other personal information. |
| R5: data | report.md §2; TRAINING_SOURCES.json | Training source, local revision evidence, retained benchmark list, preprocessing, withheld-development caveat. |
| R5: submitted approach | report.md §3 and §4 | Reference algorithm separated from regressing extension; model equations, priors, acquisition, and calibration. |
| R5: implementation / experimental setup | report.md §5 and §7 | Deployment versus development dependencies; benchmark-only and diagonal splits distinguished. |
| R5: results | report.md §6 | Local tables separate from hosted values. Hosted values explicitly secondary evidence, not freshly retrieved receipts. |
| R5: limitations | report.md §8; Appendix B | Local/hosted mismatch, sample redraw, incomplete scorer archive, grouping and data caveats; no unsupported causal claim. |
| R5: rerun instructions | report.md Appendix A; EVIDENCE.md; ARTIFACT_SHA256.json | Existing-artifact checks and staged training/evaluation commands. Full retraining not run for this draft; original historical dependency lock UNKNOWN. Complete snapshot needed to preserve pre-filter fold permutation. |
| R6–R7: release code, permitted differing data, and links | report.md draft notice, §7, Appendix B | PREPARED LOCALLY: source-only standalone repository. Intended link: https://github.com/cloudlabworks/aims-team-drift . OPEN: actual publication/access; no push performed. No additional response training dataset differs from measurement-db; gated data and all weights excluded. |
| R7: OpenReview | report.md References; REQUIREMENTS.md, R7 | Venue https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC . Form fields and accepted file type UNKNOWN; not submitted. |
| R8: all training sources disclosed | report.md §2; TRAINING_SOURCES.json; EVIDENCE.md | measurement-db final training inventory and historical excluded-source development caveat disclosed. |
| R8: AI coding assistance | report.md Acknowledgements | Explicit coding and report-drafting assistance disclosure. Exact organiser wording quoted in REQUIREMENTS.md. No claim that a separate writing-AI policy was found. |
| R9: inventory-only competition training/development | report.md §2 and §8 | Final filter inspectable; full early-development inventory compliance (including synthetic fixtures) remains UNKNOWN, not certified. |
| R10: pretrained model disclosure | report.md §2 and §4; EVIDENCE.md | Extension identifies potion-base-8M and pinned revision. Reference uses no pretrained model. |
| R11–R12: runtime and adaptation | report.md §1, §3, §4, §7 | No network inference path; only supplied labels used at test time. This is source inspection, not a new hosted execution. |
| R13: contributions acknowledged | report.md Acknowledgements and References | Organiser dataset/baseline and embedding source acknowledged; no invented collaborator. |
| R14: private feedback restrictions | report.md §6.2 | Aggregate supplied scores only; no private test records, anonymous-ID identification, or private feedback export. |
| Task: honest reference and regression | report.md Abstract, §6, §8 | v2 retained; v3 regression and local validation mismatch plainly stated. |
| Task: numeric provenance | report.md HTML comments; EVIDENCE.md | Numerical results/configuration values have repository source comments. Hosted figures originate in a clearly labelled secondary transcription. |
| Task: requested inputs | EVIDENCE.md | REPORT.md, logs, both submission/dev trees, NOTES_scope.txt, task record, and Git history inspected. Actual results path is aims/work/RESULTS.md; aims/RESULTS.md absent. |
| Task: draft only and privacy | Entire report directory | No submission, release, upload, external contact, financial content, unrelated employer material, or additional personal data. |

## Length audit

Count method: remove HTML source comments, then count whitespace-separated tokens. This is a transparent approximate word count, not a linguistic word tokenizer or a layout estimate. Main text includes the title/status block, abstract, sections, tables, equations, and Acknowledgements; references and appendices are excluded. Code tokens are included in the whole-document count.

Final count: **2,551 main-text tokens** (including Acknowledgements); **3,393 whole-document tokens**. These are whitespace-based approximate words. **Rendered pages: 6 main text / 10 total**, measured after NeurIPS conversion; see `PDF_CHECK.json`. The organiser specifies approximately 4–8 main-text pages, excluding references and appendices, with no stated word limit. The earlier whitespace counts above describe the input draft, not the revised release-status prose; actual pagination is authoritative.

## Preparation still required before any eventual submission

- Template/rendering/page checks are complete locally; inspect the PDF during author review.
- Publish the prepared source-only release and verify the report link; do not redistribute the gated data, derived per-item tables or weights.
- Retrieve the historical hosted scoring exports if available; preserve the secondary-evidence qualifier until then.
- Resolve the full historical development-data inventory audit and remaining reproduction uncertainties.
- Check live OpenReview form requirements and accepted upload format.
- Obtain author review of the content and final submission selection. This task does not assert that author review has occurred or initiate a submission workflow.

The complete UNKNOWN list is in REQUIREMENTS.md. No unresolved item above was silently treated as satisfied.
