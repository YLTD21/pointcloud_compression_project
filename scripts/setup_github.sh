# scripts/setup_github.sh
#!/bin/bash

echo "=== GitHub仓库设置 ==="

# 检查是否在项目目录
if [ ! -d ".git" ]; then
    echo "错误: 当前目录不是Git仓库"
    exit 1
fi

# 设置用户信息
read -p "请输入GitHub用户名: " username
read -p "请输入GitHub邮箱: " email

git config --global user.name "$username"
git config --global user.email "$email"

# 解决安全目录问题
git config --global --add safe.directory /media/yml/share/pointcloud_compression_project

# 设置远程仓库
read -p "请输入GitHub仓库URL (例如: https://github.com/YLTD21/pointcloud_compression_project.git): " repo_url

if [ -n "$repo_url" ]; then
    git remote remove origin 2>/dev/null
    git remote add origin "$repo_url"
    echo "远程仓库设置为: $repo_url"
fi

# 重命名分支为main（如果当前是master）
current_branch=$(git branch --show-current)
if [ "$current_branch" = "master" ]; then
    git branch -M main
    echo "分支重命名为: main"
fi

echo "=== GitHub设置完成 ==="
echo "当前配置:"
git config --list | grep -E "(user.name|user.email|remote.origin.url)"