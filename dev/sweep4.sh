#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
run(){ S=$1; tag=$2; shift; shift; echo "== $tag S=$S"; .venv/bin/python dev/eval_model.py --S $S --tag $tag "$@" 2>&1 | grep -E "RESULT|rror"; }
run 3 best
run 10i best
run 10 d2.5 --sd '{"d":2.5}'
run 10 d2.5_p0.6 --sd '{"d":2.5,"delta":0.6}'
run 10 d2.5_p1.0_b1.0 --sd '{"d":2.5,"delta":1.0,"beta":1.0}'
