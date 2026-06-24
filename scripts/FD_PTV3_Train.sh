#!/bin/bash

# ================================================================
# FD_PTV3 联邦学习训练启动脚本
# ================================================================
# 用法:
#   bash scripts/FD_PTV3_Train.sh -d s3dis -c FDPTV3-semseg-pt-v3m1-1-rpe -n my_exp
#
# 断点续传:
#   bash scripts/FD_PTV3_Train.sh -d s3dis -c FDPTV3-semseg-pt-v3m1-1-rpe -n my_exp -r true
# ================================================================

cd "$(dirname "$(dirname "$0")")" || exit
ROOT_DIR=$(pwd)
PYTHON=python

DATASET=s3dis
CONFIG="None"
EXP_NAME=debug
RESUME=false
GPU=1

while getopts "p:d:c:n:g:r:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_NAME=$OPTARG ;;
    g) GPU=$OPTARG ;;
    r) RESUME=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG"; exit 1 ;;
  esac
done

# ---- 默认 GPU 数量 ----
if [ "$GPU" = "None" ]; then
  GPU=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
fi

echo "=============================================="
echo "  FD_PTV3 — Flower-based Federated Learning"
echo "=============================================="
echo "  Experiment : $EXP_NAME"
echo "  Dataset    : $DATASET"
echo "  Config     : $CONFIG"
echo "  GPU        : $GPU"
echo "  Resume     : $RESUME"
echo "  Root       : $ROOT_DIR"
echo "=============================================="

# ---- 路径 ----
EXP_DIR=exp/${DATASET}/${EXP_NAME}
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

echo ""
echo " =========> CREATE EXP DIR <========="
echo " Experiment dir: $ROOT_DIR/$EXP_DIR"

if [ "$RESUME" = "true" ]; then
  if [ ! -f "$EXP_DIR/resume_state.json" ]; then
    echo " [ERROR] 断点文件不存在: $EXP_DIR/resume_state.json"
    echo " 请确认实验名正确，或先执行首次训练"
    exit 1
  fi
  echo " [断点续传] 检测到 resume_state.json，从中断处继续"
else
  mkdir -p "$EXP_DIR"
  echo " [新实验] 目录已创建"
fi

echo " 加载配置: $CONFIG_DIR"
echo ""

# ---- CUDA 显存优化 ----
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

echo " =========> RUN TRAINING <========="

$PYTHON -m FD_PTV3.fd_train \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options save_path="$EXP_DIR" resume="$RESUME"

TRAIN_EXIT=$?

echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
  echo " =========> TRAINING COMPLETED <========="
  echo " 模型: $EXP_DIR/final_model.pth"
  echo " 测试结果: $EXP_DIR/final_test/"
  echo " 日志: $EXP_DIR/federated_training.log"
else
  echo " =========> TRAINING FAILED (exit code: $TRAIN_EXIT) <========="
  exit $TRAIN_EXIT
fi
