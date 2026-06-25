#!/bin/bash

# ================================================================
# FDPTV3_refactor 联邦学习模型测试脚本
# ================================================================
# 用法:
#   bash scripts/FDPTV3_refactor_Test.sh -d s3dis -c FDPTV3_refactor-example-fedavg-standard -n my_exp
# ================================================================

cd "$(dirname "$(dirname "$0")")" || exit
ROOT_DIR=$(pwd)
PYTHON=python

DATASET=s3dis
CONFIG="FDPTV3_refactor-example-fedavg-standard"
EXP_NAME=debug
WEIGHT="None"
GPU=1

while getopts "p:d:c:n:w:g:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_NAME=$OPTARG ;;
    w) WEIGHT=$OPTARG ;;
    g) GPU=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG"; exit 1 ;;
  esac
done

EXP_DIR=exp/${DATASET}/${EXP_NAME}
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

if [ ! -f "$CONFIG_DIR" ]; then
  echo " [ERROR] 配置文件不存在: $CONFIG_DIR"
  exit 1
fi

if [ "$WEIGHT" = "None" ]; then
  if [ -f "$EXP_DIR/final_model.pth" ]; then
    WEIGHT="$EXP_DIR/final_model.pth"
    echo " [Auto] 使用最终模型: $WEIGHT"
  elif [ -f "$EXP_DIR/Fed_model/global_last.pth" ]; then
    WEIGHT="$EXP_DIR/Fed_model/global_last.pth"
    echo " [Auto] 使用最新全局模型: $WEIGHT"
  else
    echo " [ERROR] 未找到权重文件，请用 -w 指定路径"
    exit 1
  fi
fi

echo "=============================================="
echo "  FDPTV3_refactor — Model Testing"
echo "=============================================="
echo "  Experiment : $EXP_NAME"
echo "  Config     : $CONFIG"
echo "  Weight     : $WEIGHT"
echo "  GPU        : $GPU"
echo "  Root       : $ROOT_DIR"
echo "=============================================="

$PYTHON -m FDPTV3_refactor.fd_test \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --weight "$WEIGHT" \
  --save-path "$EXP_DIR/test_result"

TEST_EXIT=$?

echo ""
if [ $TEST_EXIT -eq 0 ]; then
  echo " =========> TESTING COMPLETED <========="
  echo " 结果: $EXP_DIR/test_result/"
else
  echo " =========> TESTING FAILED (exit code: $TEST_EXIT) <========="
  exit $TEST_EXIT
fi