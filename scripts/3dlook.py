import open3d as o3d
# 读取点云文件（支持PLY、PCD等格式，替换为你的文件路径）
pcd = o3d.io.read_point_cloud("data/processed_dataset/seq_00/000027.npy")
# 打印点云信息（可选，查看点数等基础数据）
print("点云信息：", pcd)
# 可视化点云，窗口支持缩放、旋转操作
o3d.visualization.draw_geometries([pcd], window_name="Open3D 点云可视化")
