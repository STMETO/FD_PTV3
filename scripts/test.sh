# #!/bin/sh

# cd $(dirname $(dirname "$0")) || exit
# PYTHON=python

# TEST_CODE=test.py

# DATASET=s3dis
# CONFIG="None"
# EXP_NAME=debug
# WEIGHT=model_best
# NUM_GPU=None
# NUM_MACHINE=1
# DIST_URL="auto"

# while getopts "p:d:c:n:w:g:m:" opt; do
#   case $opt in
#     p)
#       PYTHON=$OPTARG
#       ;;
#     d)
#       DATASET=$OPTARG
#       ;;
#     c)
#       CONFIG=$OPTARG
#       ;;
#     n)
#       EXP_NAME=$OPTARG
#       ;;
#     w)
#       WEIGHT=$OPTARG
#       ;;
#     g)
#       NUM_GPU=$OPTARG
#       ;;
#     m)
#       NUM_MACHINE=$OPTARG
#       ;;
#     \?)
#       echo "Invalid option: -$OPTARG"
#       ;;
#   esac
# done

# if [ "${NUM_GPU}" = 'None' ]
# then
#   NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
# fi

# echo "Experiment name: $EXP_NAME"
# echo "Python interpreter dir: $PYTHON"
# echo "Dataset: $DATASET"
# echo "GPU Num: $NUM_GPU"
# echo "Machine Num: $NUM_MACHINE"

# if [ -n "$SLURM_NODELIST" ]; then
#   MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
#   MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
#   MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
#   DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
# fi

# echo "Dist URL: $DIST_URL"

# EXP_DIR=exp/${DATASET}/${EXP_NAME}
# MODEL_DIR=${EXP_DIR}/model
# CODE_DIR=${EXP_DIR}/code
# CONFIG_DIR=${EXP_DIR}/config.py

# if [ "${CONFIG}" = "None" ]
# then
#     CONFIG_DIR=${EXP_DIR}/config.py
# else
#     CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
# fi

# echo "Loading config in:" $CONFIG_DIR
# #export PYTHONPATH=./$CODE_DIR
# export PYTHONPATH=./
# echo "Running code in: $CODE_DIR"


# echo " =========> RUN TASK <========="
# ulimit -n 65536
# #$PYTHON -u "$CODE_DIR"/tools/$TEST_CODE \
# $PYTHON -u tools/$TEST_CODE \
#   --config-file "$CONFIG_DIR" \
#   --num-gpus "$NUM_GPU" \
#   --num-machines "$NUM_MACHINE" \
#   --machine-rank ${SLURM_NODEID:-0} \
#   --dist-url ${DIST_URL} \
#   --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth


#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
PYTHON=python

TEST_CODE=test.py

# --- 默认值 ---
DATASET=s3dis
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"
# --- 新增：用于接收完整路径的变量 ---
FULL_CONFIG_PATH=""
FULL_WEIGHT_PATH=""
CUSTOM_EXP_DIR=""

# MODIFIED: 在 getopts 中增加了 f: l: e: 三个新选项
while getopts "p:d:c:n:w:g:m:f:l:e:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    # --- 新增：处理新的路径选项 ---
    f)
      FULL_CONFIG_PATH=$OPTARG
      ;;
    l)
      FULL_WEIGHT_PATH=$OPTARG
      ;;
    e)
      CUSTOM_EXP_DIR=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]; then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

# SLURM 分布式设置 (保持不变)
if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi
echo "Dist URL: $DIST_URL"

# --- MODIFIED: 路径处理逻辑 ---
# 1. 实验目录路径 (EXP_DIR)
if [ -n "$CUSTOM_EXP_DIR" ]; then
  # 如果用户通过 -e 指定了路径，则优先使用
  EXP_DIR="$CUSTOM_EXP_DIR"
  echo "Using custom experiment directory: $EXP_DIR"
else
  # 否则，使用原来的拼接方式
  EXP_DIR=exp/${DATASET}/${EXP_NAME}
fi

# 2. 配置文件路径 (CONFIG_DIR)
if [ -n "$FULL_CONFIG_PATH" ]; then
  # 如果用户通过 -f 指定了完整路径，则优先使用
  CONFIG_DIR="$FULL_CONFIG_PATH"
else
  # 否则，使用原来的逻辑
  if [ "${CONFIG}" = "None" ]; then
    CONFIG_DIR=${EXP_DIR}/config.py
  else
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
  fi
fi

# 3. 模型权重路径 (WEIGHT_PATH)
if [ -n "$FULL_WEIGHT_PATH" ]; then
  # 如果用户通过 -l 指定了完整路径，则优先使用
  WEIGHT_PATH="$FULL_WEIGHT_PATH"
else
  # 否则，使用原来的拼接方式
  MODEL_DIR=${EXP_DIR}/model
  WEIGHT_PATH="${MODEL_DIR}"/"${WEIGHT}".pth
fi

echo "Loading config in: $CONFIG_DIR"
echo "Loading weight from: $WEIGHT_PATH"
export PYTHONPATH=./

echo " =========> RUN TASK <========="
ulimit -n 65536
$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$NUM_GPU" \
  --num-machines "$NUM_MACHINE" \
  --machine-rank ${SLURM_NODEID:-0} \
  --dist-url ${DIST_URL} \
  --options save_path="$EXP_DIR" weight="$WEIGHT_PATH" # MODIFIED: 使用新的 WEIGHT_PATH 变量