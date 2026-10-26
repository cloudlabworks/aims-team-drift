# AIMS report requirements — source record

Retrieved 2026-10-26 with web_fetch. Requested URL: https://aimslab.stanford.edu/competition ; final URL: https://aims.stanford.edu/competition . This file distinguishes organiser statements from drafting choices. The page is live and can change.

## Quoted requirements

| ID | Exact organiser quotation | Source URL |
|---|---|---|
| R1 — deadline | “Final codebase and technical report deadline: 30 October 2026 at 23:59 Anywhere on Earth (AoE).” | https://aims.stanford.edu/competition |
| R2 — selected entry | “For final evaluation, teams must submit their codebase and technical report by the submission deadline and provide their selected Codabench submission ID in the OpenReview submission form.” | https://aims.stanford.edu/competition |
| R3 — template and extent | “The technical report should use the NeurIPS template and contain approximately 4–8 pages of main text, excluding references and appendices. References and appendices have no page limit.” | https://aims.stanford.edu/competition |
| R4 — identity | “Technical reports do not need to be anonymous.” | https://aims.stanford.edu/competition |
| R5 — content and reproducibility | “The report must clearly describe the data, submitted approach, implementation and experimental setup, results, and limitations, with instructions sufficient to rerun the method.” | https://aims.stanford.edu/competition |
| R6 — release | “Consideration for an oral or poster presentation requires a technical report and release of the code used to develop and run the submission, along with any permitted training data that differ from measurement-db.” | https://aims.stanford.edu/competition |
| R7 — delivery and links | “By the submission deadline, teams must submit the technical report, including links to these released materials, through OpenReview.” The word OpenReview links to https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC . | https://aims.stanford.edu/competition |
| R8 — training disclosure and AI assistance | “Disclose all training sources in the technical report. The use of LLM-based coding assistants during development is permitted. We kindly ask participants to disclose any such use in the technical report.” | https://aims.stanford.edu/competition |
| R9 — training restrictions | “All data sources listed in the benchmark inventory are permitted for competition-specific training, development, and curation. Participants must not use data sources outside that inventory for these purposes.” | https://aims.stanford.edu/competition |
| R10 — pretrained models | “Participants may use pretrained models and must disclose them in the technical report. Any competition-specific training or fine-tuning must use only data sources listed in the benchmark inventory.” | https://aims.stanford.edu/competition |
| R11 — runtime | “Network access during scoring is limited to inference APIs from Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI, and Groq. Web search, browsing, and grounding are prohibited, including when offered through an approved provider's API.” | https://aims.stanford.edu/competition |
| R12 — permitted adaptation | “Participants carry out initial model training before submission. The submitted program may then adapt during evaluation using the supplied attributes and labels revealed within its budget.” | https://aims.stanford.edu/competition |
| R13 — shared work | “Participants are encouraged to discuss methods and share code and permitted training data across teams, with shared contributions acknowledged in the technical report.” | https://aims.stanford.edu/competition |
| R14 — private feedback | “Formative feedback may guide development, but participants must not extract or share private test data or attempt to identify the subjects or benchmarks behind anonymous feedback IDs for use in development.” | https://aims.stanford.edu/competition |

## Evaluation facts quoted for the report

Source for every quotation below: https://aims.stanford.edu/competition .

- “Brier ALC measures predictive performance for all submissions, with or without test-time adaptation; lower values indicate better performance.”
- “We measure Brier score at label budgets of 0, 1, 3, 7, 15, and 31, using the same fixed evaluation items and excluding acquisition items.”
- “We calculate Brier ALC for each subject–benchmark pair, then average the pair scores equally. Every pair has the same weight, regardless of its number of responses.”
- Displayed formula, transcribed without duplicated HTML/TeX: Brier ALC = 0.1 B₀ + 0.2 B₁ + 0.2 B₃ + 0.2 B₇ + 0.2 B₁₅ + 0.1 B₃₁.
- “For each subject–benchmark pair with at least 80 distinct items, items are randomly split 50/50 into acquisition and evaluation pools.”
- “ECE is the mean absolute gap between average predicted probabilities and observed success rates across 10 equal-width probability bins, weighted by response count. Summary ECE uses the ALC budget weights.”
- “Each formative evaluation uses a newly sampled subset of the hidden test data, capped at 1,000 unique subject–item pairs across the acquisition and evaluation pools.”
- “All label budgets within a submission use the same evaluation responses, and the automatic component of summative evaluation uses a common test subset for all teams.”
- “The primary emphasis will be on technical soundness, clarity, reproducibility, and the quality of experimental analysis. Methodological novelty will be a secondary consideration.”
- “We do not currently plan to conduct formal peer review of technical reports.”

## Interpretation and deliberately unresolved items

The extent is **approximately 4–8 main-text pages**, not an explicitly worded hard eight-page maximum. No word limit is stated. The page requests a NeurIPS template, but the fetched text does not explicitly require LaTeX, a particular source-file format, or a specific template-year package. Therefore this draft supplies report.md and does not invent a mandatory report.tex requirement. Markdown is not claimed to meet the final template requirement.

The organiser AI-use rule above concerns coding assistants and requests disclosure; it does **not** explicitly require an acknowledgements section or specify a separate generative-writing policy. The draft places a broader coding-and-report assistance disclosure in Acknowledgements as requested by the author.

- UNKNOWN: exact NeurIPS template download/version and template options intended for this competition; no specific template link appears in the fetched competition text. A supplementary probe of https://neurips.cc/Conferences/2026/PaperInformation/StyleFiles returned 404.
- UNKNOWN: accepted upload file type, source-file requirements, and additional mandatory OpenReview form fields. The public venue URL resolves, but web_fetch returned only the generic OpenReview footer, not a submission form.
- UNKNOWN: whether an additional report-writing AI policy exists outside the fetched competition page. No such additional rule is asserted.
- UNKNOWN: final typeset main-text page count; this Markdown draft is unpaginated.
- UNKNOWN: public release URL and access status for Team Drift's development code, submission artifacts, and permitted derived data.
- UNKNOWN: archival scorer definition attached to the recorded hosted `brier` and `calibration_ece` fields at the time of the historical submissions. The current page defines Brier ALC, but a historical scoring export is not in this repository.
- UNKNOWN: primary hosted scoring receipts, per-budget feedback, evaluated-pair counts, and repeat-run uncertainty for the recorded submissions; only the supplied task record and a partial earlier repository note are available here.
- UNKNOWN: causal explanation for the local-versus-hosted ranking reversal.
- UNKNOWN: original historical training-environment dependency lock and guaranteed bit-identical full retraining; the current environment can be inventoried, not retrospectively certified.
- UNKNOWN: exact hosted subject sampling, cross-subject item overlap, and treatment of graded-response source benchmarks beyond what the current competition page states.
- UNKNOWN: complete inventory-level compliance audit of early development sweeps, which included subsequently excluded withheld-release benchmarks and synthetic interface-test fixtures; the final prior's public-benchmark filter is inspectable.

## Scope

This is a draft-only preparation. Nothing has been submitted, uploaded, posted, released, or sent to organisers by this task. Do not interpret the checklist as a receipt for any of those actions.

## Local preparation addendum — 2026-10-26

The earlier UNKNOWN entries above describe the draft intake. Official 2026 template retrieval is now documented in `STYLE_SOURCE.json`; `report.pdf` uses final/non-anonymous workshop mode. `PDF_CHECK.json` records 6 main-text pages including Acknowledgements and 10 total pages. A standalone source-only release is prepared locally, with intended URL https://github.com/cloudlabworks/aims-team-drift . Actual public availability and OpenReview submission remain unperformed. No measurement-db records, derived per-item tables or model weights are included. See the release root README and RELEASE_CHECK.md for reconstruction instructions and checks.
