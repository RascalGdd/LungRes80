#!/usr/bin/env bash
set -euo pipefail

# LungRes80 training and online evaluation entry point.

PROJECT_ROOT=${PROJECT_ROOT:-/home/diandian/Diandian/DD/LungReco}
DATA_ROOT=${DATA_ROOT:-/home/diandian/Diandian/DATASETS/LungRes80}
PRETRAINED_PATH=${PRETRAINED_PATH:-/home/diandian/Diandian/DD/LungSeg/TimeSformer_divST_8x32_224_K400.pyth}
LLAVA_MODEL_PATH=${LLAVA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}
TORCHRUN_BIN=${TORCHRUN_BIN:-/home/diandian/anaconda3/envs/xlstm/bin/torchrun}
RUN_NAME=${RUN_NAME:-frame32_rfm04_run1}

EXPERIMENT_ROOT=$PROJECT_ROOT/results/LungRes80_${RUN_NAME}
BASE_ROOT=$EXPERIMENT_ROOT/base
HISTORY_ROOT=$EXPERIMENT_ROOT/history_finetune
FINAL_ROOT=$EXPERIMENT_ROOT/final_test_online
TRAIN_VIDEO_IDS=50,64,24,51,31,56,09,47,58,40,16,73,60,39,43,63,26,57,77,02,71,54,19,68,20,21,44,48,34,62,33,08,66,03,17,37,23,41,25,06,07,61,74,10,53,22,67,59,11,49

common_args=(
  --batch_size 8 --epochs 50 --start_epoch 0
  --no_auto_resume --save_ckpt_freq 1 --early_stopping_patience 3
  --select_on_val_only
  --model lungreco --pretrained_path "$PRETRAINED_PATH"
  --llava_model_path "$LLAVA_MODEL_PATH"
  --mcr_mask_ratio 0
  --gaussian_sigma 3.0 --gaussian_history_length 64
  --feature_memory_gaussian_smoothing
  --include_text_final_fusion --history_text_scale 1.0
  --llava_prompt_cache_size 4096
  --train_video_ids "$TRAIN_VIDEO_IDS"
  --use_checkpoint --opt adamw --opt_betas 0.9 0.999
  --weight_decay 0.05 --lr 1e-4 --disable_lr_scale
  --layer_decay 0.75 --warmup_epochs 0
  --mixup 0 --cutmix 0 --smoothing 0 --reprob 0 --paper_augmentation
  --data_path "$DATA_ROOT" --eval_data_path "$DATA_ROOT"
  --nb_classes 7 --data_strategy online --output_mode key_frame
  --num_frames 16 --sampling_rate 8 --data_set LungRes80 --data_fps 1fps
  --num_workers 4 --dist_eval --bf16 --enable_deepspeed
)

best_epoch() {
  python3 -c 'import json,sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(int(rows[-1]["max_val_epoch"]))' "$1"
}

find_run_dir() {
  local root=$1
  local run_dir
  run_dir=$(find "$root" -type d -name checkpoint-best -printf '%h\n' | head -n 1)
  test -n "$run_dir"
  printf '%s\n' "$run_dir"
}

cd "$PROJECT_ROOT"
test -x "$TORCHRUN_BIN"
test -d "$DATA_ROOT"
test -f "$PRETRAINED_PATH"
# Never overwrite or silently resume an earlier experiment.
test ! -e "$EXPERIMENT_ROOT"

echo "[Base training] all 50 train videos"
"$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
  downstream_phase/run_phase_training.py \
  "${common_args[@]}" \
  --transition_history_length 3 \
  --train_sample_interval 1 \
  --phase_history_mask_probability 0 --phase_history_replace_ratio 0 \
  --output_dir "$BASE_ROOT" --log_dir "$BASE_ROOT/log"

BASE_RUN_DIR=$(find_run_dir "$BASE_ROOT")
BASE_BEST_CHECKPOINT=$BASE_RUN_DIR/checkpoint-best/mp_rank_00_model_states.pt
test -f "$BASE_BEST_CHECKPOINT"
BASE_BEST_EPOCH=$(best_epoch "$BASE_RUN_DIR/log.txt")
echo "[Base training] best_epoch=$BASE_BEST_EPOCH"
echo "[Base training] checkpoint=$BASE_BEST_CHECKPOINT"

echo "[History adaptation] start"
"$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
  downstream_phase/run_phase_training.py \
  "${common_args[@]}" \
  --finetune "$BASE_BEST_CHECKPOINT" \
  --transition_history_length 32 --frame_phase_history \
  --train_sample_interval 32 \
  --prediction_history_bank --freeze_prediction_history_bank \
  --gt_train_history_bank \
  --phase_history_mask_probability 0.4 --phase_history_replace_ratio 0 \
  --prediction_history_dropout 0 --prediction_history_corruption 0 \
  --freeze_except_history_fusion \
  --output_dir "$HISTORY_ROOT" --log_dir "$HISTORY_ROOT/log"

HISTORY_RUN_DIR=$(find_run_dir "$HISTORY_ROOT")
HISTORY_BEST_CHECKPOINT=$HISTORY_RUN_DIR/checkpoint-best/mp_rank_00_model_states.pt
test -f "$HISTORY_BEST_CHECKPOINT"
HISTORY_BEST_EPOCH=$(best_epoch "$HISTORY_RUN_DIR/log.txt")
echo "[History adaptation] best_epoch=$HISTORY_BEST_EPOCH"
echo "[History adaptation] checkpoint=$HISTORY_BEST_CHECKPOINT"

echo "[Final test] causal online inference"
"$TORCHRUN_BIN" --standalone --nproc_per_node=2 \
  downstream_phase/run_phase_training.py \
  "${common_args[@]}" \
  --finetune "$HISTORY_BEST_CHECKPOINT" \
  --eval_lungreco_test \
  --transition_history_length 32 --frame_phase_history \
  --train_sample_interval 32 \
  --prediction_history_bank --freeze_prediction_history_bank \
  --phase_history_mask_probability 0 --phase_history_replace_ratio 0 \
  --prediction_history_dropout 0 --prediction_history_corruption 0 \
  --output_dir "$FINAL_ROOT" --log_dir "$FINAL_ROOT/log"

FINAL_PREDICTION_DIR=$(find "$FINAL_ROOT" -type d -path '*/test_predictions/final' | head -n 1)
test -n "$FINAL_PREDICTION_DIR"
test "$(wc -l < "$FINAL_PREDICTION_DIR/unrelaxed.jsonl")" -eq 59474
test -f "$FINAL_PREDICTION_DIR/metrics_strict.json"
test -f "$FINAL_PREDICTION_DIR/metrics_relaxed_k5.json"

echo "[Done] base_best_epoch=$BASE_BEST_EPOCH"
echo "[Done] history_best_epoch=$HISTORY_BEST_EPOCH"
echo "[Done] checkpoint=$HISTORY_BEST_CHECKPOINT"
echo "[Done] test_predictions=$FINAL_PREDICTION_DIR/unrelaxed.jsonl"
echo "[Done] test_strict=$FINAL_PREDICTION_DIR/metrics_strict.json"
echo "[Done] test_relaxed_k5=$FINAL_PREDICTION_DIR/metrics_relaxed_k5.json"
