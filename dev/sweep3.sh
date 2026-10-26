#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
run(){ tag=$1; shift; echo "== $tag"; .venv/bin/python dev/eval_model.py --S 10 --tag $tag "$@" 2>&1 | grep -E "RESULT|rror"; }
run var_du0.0 --opts '{"var":1.0,"du":0.0}'
run var_du0.5 --opts '{"var":1.0,"du":0.5}'
run var_du1.2 --opts '{"var":1.0,"du":1.2}'
run var_du0.5_half --opts '{"var":0.5,"du":0.5}'
run ss0 --opts '{"ss":0.0}'
run si0 --opts '{"si":0.0}'
run ss1.5 --opts '{"ss":1.5}'
run si1.5 --opts '{"si":1.5}'
