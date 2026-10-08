#!/usr/bin/env bash
# Runs the NF-UNSW-NB15-v2 configs one after another, after the 20q ion-trap job.
LOG20=/d/Dev/ai-ml/reservoir-computing/ion_traps/run_20q_ring.log
until grep -qE "\[save\] results|Traceback" "$LOG20"; do sleep 30; done
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
P=/c/Users/binh1/miniconda3/envs/py3.10_pytorch/python.exe
cd /d/Dev/ai-ml/qorc
for c in nfunsw_dedup_supervised nfunsw_dedup_classical nfunsw_dedup_novelty nfunsw_classical nfunsw_supervised; do
  echo "=== $(date +%T) start $c"
  $P -m elements.run configs/$c.yaml > runs/$c.out 2>&1
  echo "=== $(date +%T) end $c (exit $?)"
done
