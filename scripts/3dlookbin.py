import numpy as np
import open3d as o3d

# 1. 读取.bin文件（替换为你的.bin文件路径）
bin_path = "/path/to/your/semantic_kitti/velodyne/000000.bin"
# 解析：每个点4个float32值（x,y,z,intensity），总字节数=点数×4×4
points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # 形状为[N,4]

# 2. 提取x,y,z坐标（忽略反射强度）
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 只取前3列（x,y,z）

# 3. 可视化（支持旋转、缩放、平移）
o3d.visualization.draw_geometries([pcd], window_name="SemanticKITTI 点云")
