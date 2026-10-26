#!/bin/bash
set -euo pipefail
# Inner (out-of-fold) tuning grid for the v3 embedding term: for every config, OOF runs f x j!=f on S in {10,3,10i}.
cd "$(dirname "$0")/.."
run() { tag=$1; opts=$2; sd=$3
  [ -f work/res3/${tag}_S10i_oof_f4_j3.pkl ] && { echo "skip $tag"; return; }
  for S in 10 3 10i; do .venv/bin/python -I dev3/eval3.py --model v3 --tag $tag --S $S --mode oof --opts "$opts" --sd "$sd" 2>&1 | grep -v Warning | tail -1; done
  echo "done $tag $(date +%H:%M:%S)"; }
for d in 1.5 0.75; do
  run t_k0_d$d '{"k":0}' "{\"d\":$d}"
  for k in 8 16 32; do for su in 0.2 0.5 1.0; do
    run t_k${k}_su${su}_d$d "{\"k\":$k,\"su\":$su}" "{\"d\":$d}"
  done; done
done
