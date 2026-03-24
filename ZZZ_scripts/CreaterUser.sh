#!/bin/bash

echo "=== 创建新用户 ==="

# 获取用户名
read -p "请输入要创建的用户名: " USERNAME

# 检查用户名是否为空
if [ -z "$USERNAME" ]; then
    echo "✗ 用户名不能为空"
    exit 1
fi

# 检查用户是否已存在
if id "$USERNAME" &>/dev/null; then
    echo "✗ 用户 $USERNAME 已存在"
    exit 1
fi

echo "正在创建用户: $USERNAME"

# 创建用户
sudo useradd -m -s /bin/bash "$USERNAME"
if [ $? -eq 0 ]; then
    echo "✓ 用户 $USERNAME 创建成功"
else
    echo "✗ 用户创建失败"
    exit 1
fi

# 设置密码
echo "请为用户 $USERNAME 设置密码："
sudo passwd "$USERNAME"

# 设置权限
sudo chown -R "$USERNAME":"$USERNAME" "/home/$USERNAME"

echo ""
echo "=== 创建完成 ==="
echo "现在可以使用以下命令登录："
echo "ssh $USERNAME@服务器IP"