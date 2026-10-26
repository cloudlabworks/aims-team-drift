#!/bin/bash
set -euo pipefail
# Second inner grid: smaller embedding-prior sd (best configs sat at the su=0.2 edge) and a tighter item-offset prior.
cd "$(dirname "$0")/.."
run() { tag=$1; opts=$2; sd=$3
  [ -f work/res3/${tag}_S10i_oof_f4_j3.pkl ] && { echo "skip $tag"; return; }
  for S in 10 3 10i; do .venv/bin/python -I dev3/eval3.py --model v3 --tag $tag --S $S --mode oof --opts "$opts" --sd "$sd" 2>&1 | grep -v Warning | tail -1; done
  echo "done $tag $(date +%H:%M:%S)"; }
for su in 0.05 0.1 0.15; do for k in 16 32; do
  run t_k${k}_su${su}_d1.5 "{\"k\":$k,\"su\":$su}" '{"d":1.5}'
done; done
run t_k0_d0.75 '{"k":0}' '{"d":0.75}'
for su in 0.1 0.2; do run t_k16_su${su}_d0.75 "{\"k\":16,\"su\":$su}" '{"d":0.75}'; done
