#!/bin/bash

# ==================== 配置 ====================
SERVER_USER="xiaoyu"
SERVER_IP="10.30.33.79"
BASE_DIR="/mnt/data2/xiaoyu_data"
EXP_DIR="exp_data"
TENSORBOARD_PATH="/mnt/data2/xiaoyu_data/conda_envs/FDPTV3/bin/tensorboard"
# =============================================

if [ $# -ne 2 ]; then
    echo "❌ 用法：./tb.sh 端口号 实验目录"
    echo "示例：./tb.sh 8090 251103_FedMarkovAvg_r100_u3_e9_bs03"
    exit 1
fi

PORT=$1
EXP_NAME=$2

# 构建完整路径
if [[ "$EXP_NAME" == /* ]]; then
    FULL_PATH="$EXP_NAME"
else
    FULL_PATH="$BASE_DIR/$EXP_DIR/$EXP_NAME"
fi

echo "🔧 配置信息："
echo "  - 端口: $PORT"
echo "  - 实验: $EXP_NAME"
echo "  - 路径: $FULL_PATH"
echo "  - 服务器: $SERVER_USER@$SERVER_IP"
echo ""

# 分步执行，便于调试
echo "1. 检查远程目录..."
ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "
    if [ ! -d '$FULL_PATH' ]; then
        echo '❌ 目录不存在: $FULL_PATH'
        exit 1
    fi
    echo '✅ 目录存在'
"

echo "2. 清理旧进程..."
ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "
    pkill -f 'tensorboard.*$PORT' 2>/dev/null
    sleep 2
    echo '✅ 清理完成'
"

echo "3. 启动TensorBoard..."
ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "
    echo '启动命令: $TENSORBOARD_PATH --logdir=\"$FULL_PATH\" --port $PORT --bind_all'
    $TENSORBOARD_PATH --logdir=\"$FULL_PATH\" --port $PORT --bind_all > /tmp/tensorboard_$PORT.log 2>&1 &
    TB_PID=\$!
    echo 'TensorBoard PID: '\$TB_PID'
    sleep 8
    
    echo '检查进程状态:'
    if ps -p \$TB_PID > /dev/null; then
        echo '✅ 进程运行中'
    else
        echo '❌ 进程已退出'
        echo '最后日志:'
        tail -20 /tmp/tensorboard_$PORT.log
    fi
    
    echo '检查端口:'
    netstat -tulpn | grep ':$PORT' || echo '端口未监听'
    
    echo 'TensorBoard日志:'
    tail -10 /tmp/tensorboard_$PORT.log
" &

echo "4. 等待TensorBoard启动..."
sleep 10

echo "5. 建立SSH隧道..."
echo "📌 本地访问: http://localhost:$PORT"
echo "💡 按 Ctrl+C 停止"

ssh -L $PORT:localhost:$PORT -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP