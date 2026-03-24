#!/bin/bash

# 切换到项目根目录
cd $(dirname "$0")/.. || exit
PYTHON=python

# 默认参数
CONFIG_FILE=""
SAVE_PATH=""
WEIGHT_FILE=""
GPU="None"
OPTIONS=""

# 解析命令行参数
while getopts "p:c:s:w:g:o:" opt; do
  case $opt in
    p)
      PYTHON="$OPTARG"
      ;;
    c)
      CONFIG_FILE="$OPTARG"
      ;;
    s)
      SAVE_PATH="$OPTARG"
      ;;
    w)
      WEIGHT_FILE="$OPTARG"
      ;;
    g)
      GPU="$OPTARG"
      ;;
    o)
      OPTIONS="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

# 参数检查
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file must be specified with -c"
    exit 1
fi
if [ -z "$WEIGHT_FILE" ]; then
    echo "Error: Weight file must be specified with -w"
    exit 1
fi
if [ -z "$SAVE_PATH" ]; then
    SAVE_PATH="./test_result"
fi
if [ "$GPU" = "None" ]; then
    GPU=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
fi

echo "Python interpreter: $PYTHON"
echo "Config file: $CONFIG_FILE"
echo "Weight file: $WEIGHT_FILE"
echo "Save path: $SAVE_PATH"
echo "GPU Num: $GPU"
if [ ! -z "$OPTIONS" ]; then
    echo "Additional options: $OPTIONS"
fi

# 确保 Python 能找到 pointcept
export PYTHONPATH=./

echo " =========> RUN TASK <========="
if [ -z "$OPTIONS" ]; then
    $PYTHON -u FDPTV3/FDPTV_test.py \
      --config-file "$CONFIG_FILE" \
      --num-gpus "$GPU" \
      --weight "$WEIGHT_FILE" \
      --save-path "$SAVE_PATH"
else
    $PYTHON -u FDPTV3/FDPTV_test.py \
      --config-file "$CONFIG_FILE" \
      --num-gpus "$GPU" \
      --weight "$WEIGHT_FILE" \
      --save-path "$SAVE_PATH" \
      --options $OPTIONS
fi