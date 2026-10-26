#!/bin/bash
set -euo pipefail
# Chosen config (inner selection: k=32, su=0.15, d sd=1.5): embedding-only outer runs, plus raw-logit (clip 1e-6) OOF + outer runs for calibration.
cd "$(dirname "$0")/.."
O='{"k":32,"su":0.15}'; SD='{"d":1.5}'
for S in 10 3 10i; do
  .venv/bin/python -I dev3/eval3.py --model v3 --tag v3emb --S $S --opts "$O" --sd "$SD" 2>&1 | grep -v Warning | tail -1
  .venv/bin/python -I dev3/eval3.py --model v3 --tag v3raw --S $S --mode oof --opts "$O" --sd "$SD" --clip 1e-6 2>&1 | grep -v Warning | tail -1
  .venv/bin/python -I dev3/eval3.py --model v3 --tag v3raw --S $S --mode outer --opts "$O" --sd "$SD" --clip 1e-6 2>&1 | grep -v Warning | tail -1
  echo "done S=$S $(date +%H:%M:%S)"
done
