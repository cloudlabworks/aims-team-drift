#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for S in 10 3 10i; do .venv/bin/python dev/eval_model.py --S $S --tag final 2>&1 | grep -E "RESULT|rror"; done
.venv/bin/python dev/train_final.py 1e-2 1e-2 0.5 2>&1 | tail -2
