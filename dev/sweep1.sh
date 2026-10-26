#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
run(){ tag=$1; shift; echo "== $tag"; .venv/bin/python dev/eval_model.py --S 10 --tag $tag "$@" 2>&1 | grep -E "RESULT|Error|error"; }
run beta1.0 --sd '{"beta":1.0}'
run beta2.5 --sd '{"beta":2.5}'
run delta0.4 --sd '{"delta":0.4}'
run delta1.5 --sd '{"delta":1.5}'
run d0.5 --sd '{"d":0.5}'
run d1.5 --sd '{"d":1.5}'
run g0.15 --sd '{"g":0.15}'
run g0.7 --sd '{"g":0.7}'
run temp0.9 --temp 0.9
run temp1.15 --temp 1.15
