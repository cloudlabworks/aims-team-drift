#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
run(){ tag=$1; shift; echo "== $tag"; .venv/bin/python dev/eval_model.py --S 10 --tag $tag "$@" 2>&1 | grep -E "RESULT|rror"; }
run kern2 --model dev/variants/model_k.py --opts '{"c":6.0,"kappa":0.3,"q":3,"s0":0.4}'
run lam_i3e-3 --lam_i 3e-3 --lam_s 1e-2
run lam_i1e-1 --lam_i 1e-1 --lam_s 1e-2
run lam_s1e-3 --lam_i 1e-2 --lam_s 1e-3
run lam_s1e-1 --lam_i 1e-2 --lam_s 1e-1
run alpha0 --alpha 0.0
run alpha1 --alpha 1.0
run drop3 --drop 0.3
