import open3d as o3d
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def visualize_pointcloud(file_path):
    """
    可视化单个点云文件
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 根据文件扩展名选择加载方式
    if file_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(file_path)
    elif file_path.endswith('.npy'):
        data = np.load(file_path, allow_pickle=True).item()
        points = data['points']
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 如果有颜色信息，可以设置
        if 'labels' in data:
            labels = data['labels']
            colors = np.zeros((len(points), 3))
            colors[labels == 253] = [1, 0, 0]  # 车辆-红色
            colors[labels == 252] = [0, 1, 0]  # 行人-绿色
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("不支持的文件格式，请使用.pcd或.npy文件")
        return

    # 打印点云信息
    print(f"点云数量: {len(pcd.points)}")

    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name="点云可视化")


def main():
    """
    主函数：提供交互式选择点云文件并可视化
    """
    # 定义可能的数据目录
    data_dirs = [
        project_root / "data" / "processed_dataset",
        project_root / "data" / "high_value_dataset",
        project_root / "results"
    ]

    # 收集所有点云文件
    pointcloud_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            pcd_files = list(data_dir.rglob("*.pcd"))
            npy_files = list(data_dir.rglob("*.npy"))
            pointcloud_files.extend(pcd_files)
            pointcloud_files.extend(npy_files)

    if not pointcloud_files:
        print("未找到任何点云文件，请先运行数据处理流程。")
        return

    # 显示文件列表
    print("找到以下点云文件:")
    for i, file_path in enumerate(pointcloud_files):
        print(f"{i + 1}: {file_path}")

    # 用户选择
    try:
        choice = int(input("请选择要可视化的文件编号 (输入0退出): "))
        if choice == 0:
            return
        if choice < 1 or choice > len(pointcloud_files):
            print("无效选择")
            return
        selected_file = pointcloud_files[choice - 1]
        print(f"正在加载: {selected_file}")
        visualize_pointcloud(str(selected_file))
    except ValueError:
        print("请输入有效数字")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()