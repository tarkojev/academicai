#!/usr/bin/env bash

set -u
set -o pipefail

SUPPORT_SEED=123
PYTHON_BIN=python
SCRIPT_NAME=student.py
OUT_DIR="runs"

MODELS=(
  "distilbert-base-uncased"
  "bert-base-uncased"
  "t5-small"
)

FS_LIST=(10 100 1000)
TRAIN_SEEDS=(3 4 5 6 7 8 9)

EXPERIMENTS=(
  "0.1 0.1"
  "0.1 0.3"
  "0.1 0.5"
  "0.1 0.7"
  "0.1 0.9"
  "0.5 0.1"
  "0.5 0.3"
  "0.5 0.5"
  "0.5 0.7"
  "0.5 0.9"
  "1 0.1"
  "1 0.3"
  "1 0.5"
  "1 0.7"
  "1 0.9"
  "1.5 0.1"
  "1.5 0.3"
  "1.5 0.5"
  "1.5 0.7"
  "1.5 0.9"
  "2 0.1"
  "2 0.3"
  "2 0.5"
  "2 0.7"
  "2 0.9"
  "2.5 0.1"
  "2.5 0.3"
  "2.5 0.5"
  "2.5 0.7"
  "2.5 0.9"
  "3 0.1"
  "3 0.3"
  "3 0.5"
  "3 0.7"
  "3 0.9"
)

get_run_dir() {
  local model="$1"
  local fs="$2"
  local support_seed="$3"
  local train_seed="$4"
  local tau="$5"
  local alpha="$6"

  local run_name="student_${model}_fs${fs}_supp${support_seed}_seed${train_seed}_tau${tau}_a${alpha}"
  run_name="${run_name//\//_}"
  echo "${OUT_DIR}/${run_name}"
}

is_run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/meta.json" ]]
}

RUN_ITEMS=()

# Build full ordered list of runs
for FS in "${FS_LIST[@]}"; do
  for SEED in "${TRAIN_SEEDS[@]}"; do
    ENSEMBLE_DIR="runs/ensemble_fs${FS}_supp${SUPPORT_SEED}_seed${SEED}"

    if [[ ! -d "$ENSEMBLE_DIR" ]]; then
      echo "Skipping missing ensemble dir in planning: $ENSEMBLE_DIR"
      continue
    fi

    for MODEL in "${MODELS[@]}"; do
      for EXP in "${EXPERIMENTS[@]}"; do
        read -r TAU ALPHA <<< "$EXP"
        RUN_ITEMS+=("${FS}|${SEED}|${MODEL}|${TAU}|${ALPHA}|${ENSEMBLE_DIR}")
      done
    done
  done
done

TOTAL=${#RUN_ITEMS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No runnable experiments found."
  exit 0
fi

LAST_DONE=-1

for ((i=0; i<TOTAL; i++)); do
  IFS='|' read -r FS SEED MODEL TAU ALPHA ENSEMBLE_DIR <<< "${RUN_ITEMS[$i]}"
  RUN_DIR="$(get_run_dir "$MODEL" "$FS" "$SUPPORT_SEED" "$SEED" "$TAU" "$ALPHA")"

  if is_run_complete "$RUN_DIR"; then
    LAST_DONE=$i
  fi
done

if [[ "$LAST_DONE" -lt 0 ]]; then
  START_INDEX=0
  echo "No completed runs found. Starting from the beginning."
else
  START_INDEX=$((LAST_DONE - 1))
  if [[ "$START_INDEX" -lt 0 ]]; then
    START_INDEX=0
  fi
  echo "Last completed run index: $LAST_DONE"
  echo "Restarting from index: $START_INDEX (n-1 safety restart)"
fi

EXECUTED=0
FAILED=0

for ((i=START_INDEX; i<TOTAL; i++)); do
  IFS='|' read -r FS SEED MODEL TAU ALPHA ENSEMBLE_DIR <<< "${RUN_ITEMS[$i]}"
  RUN_DIR="$(get_run_dir "$MODEL" "$FS" "$SUPPORT_SEED" "$SEED" "$TAU" "$ALPHA")"

  echo "=================================================="
  echo "Index          = $i / $((TOTAL - 1))"
  echo "Run dir        = $RUN_DIR"
  echo "model          = $MODEL"
  echo "fs             = $FS"
  echo "support_seed   = $SUPPORT_SEED"
  echo "train_seed     = $SEED"
  echo "ensemble_dir   = $ENSEMBLE_DIR"
  echo "tau            = $TAU"
  echo "alpha          = $ALPHA"
  echo "=================================================="

  rm -rf "$RUN_DIR"

  if $PYTHON_BIN "$SCRIPT_NAME" \
    --student_model "$MODEL" \
    --support_seed "$SUPPORT_SEED" \
    --n_per_class "$FS" \
    --train_seed "$SEED" \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --tau "$TAU" \
    --alpha "$ALPHA"; then
    EXECUTED=$((EXECUTED + 1))
  else
    echo "[FAIL] model=$MODEL fs=$FS seed=$SEED tau=$TAU alpha=$ALPHA"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "Finished."
echo "Executed : $EXECUTED"
echo "Failed   : $FAILED"