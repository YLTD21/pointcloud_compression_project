# scripts/debug_extracted_objects.py
import open3d as o3d
import numpy as np
import time
from pathlib import Path
import sys


def debug_extracted_objects():
    """调试提取的点云数据"""
    print("=== 提取对象点云调试 ===")

    # 检查序列00的提取对象
    seq_dir = Path("../data/processed_dataset/seq_00")

    if not seq_dir.exists():
        print(f"错误：目录不存在 {seq_dir}")
        return

    pcd_files = sorted(seq_dir.glob("*.pcd"))
    print(f"找到 {len(pcd_files)} 个PCD文件")

    if not pcd_files:
        print("没有找到PCD文件")
        return

    # 检查前10个文件
    for i, pcd_file in enumerate(pcd_files[:10]):
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points = np.asarray(pcd.points)

            print(f"\n文件: {pcd_file.name}")
            print(f"  点数: {len(points)}")

            if len(points) > 0:
                print(f"  坐标范围:")
                print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

                # 测试显示这一帧
                test_display_single_frame(pcd_file, f"测试帧 {i}")
            else:
                print("  警告: 空点云")

        except Exception as e:
            print(f"  错误: {e}")


def test_display_single_frame(file_path, window_name="测试"):
    """测试显示单个点云帧"""
    try:
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print("  → 空点云，跳过显示")
            return False

        print(f"  → 显示 {len(points)} 个点...")

        # 创建新的可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=window_name,
            width=1000,
            height=800,
            left=100,
            top=100
        )

        # 创建点云对象
        test_pcd = o3d.geometry.PointCloud()
        test_pcd.points = o3d.utility.Vector3dVector(points)

        # 设置颜色
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # 创建高对比度颜色
            colors = np.ones((len(points), 3))
            if len(points) == 1:
                colors[0] = [1, 1, 0]  # 黄色
            else:
                colors[:, 0] = 1  # 红色
                colors[:, 1] = 0.5
                colors[:, 2] = 0

        test_pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(test_pcd)

        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 20.0  # 很大的点
        render_option.background_color = np.array([0, 0, 0])  # 黑色背景

        # 设置视角
        view_control = vis.get_view_control()

        if len(points) > 0:
            # 计算点云中心
            center = np.mean(points, axis=0)
            print(f"  → 点云中心: {center}")

            # 设置视角看向中心
            view_control.set_lookat(center)
            view_control.set_front([0, 0, -1])  # 从正前方看
            view_control.set_up([0, -1, 0])  # Y轴向上

            # 根据点云大小设置缩放
            max_range = np.max(np.ptp(points, axis=0))
            if max_range > 0:
                zoom_level = 1.0 / max_range
            else:
                zoom_level = 0.1

            view_control.set_zoom(min(zoom_level, 10.0))

        print("  → 显示窗口中... (关闭窗口继续)")
        vis.run()  # 阻塞，直到窗口关闭
        vis.destroy_window()

        return True

    except Exception as e:
        print(f"  → 显示失败: {e}")
        return False


def check_system_environment():
    """检查系统环境"""
    print("=== 系统环境检查 ===")
    print(f"Open3D版本: {o3d.__version__}")
    print(f"NumPy版本: {np.__version__}")

    # 测试Open3D基本功能
    try:
        test_pcd = o3d.geometry.PointCloud()
        test_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        test_pcd.points = o3d.utility.Vector3dVector(test_points)
        test_pcd.paint_uniform_color([1, 0, 0])

        print("Open3D基本功能测试: 正常")
        return True
    except Exception as e:
        print(f"Open3D基本功能测试失败: {e}")
        return False


def create_simple_test_player():
    """创建一个简单的测试播放器"""
    print("=== 简单测试播放器 ===")

    # 创建一些测试点
    test_points = []
    for i in range(100):
        x = np.random.randn() * 5
        y = np.random.randn() * 5
        z = np.random.randn() * 2
        test_points.append([x, y, z])

    test_points = np.array(test_points, dtype=np.float32)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="测试播放器", width=1000, height=800)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(test_points)
    pcd.paint_uniform_color([0, 1, 0])  # 绿色

    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    render_option.background_color = np.array([0.1, 0.1, 0.3])

    print("显示测试点云... (关闭窗口继续)")
    vis.run()
    vis.destroy_window()

    print("测试完成!")


if __name__ == "__main__":
    print("点云显示调试工具")
    print("1. 检查系统环境")
    print("2. 调试提取的点云数据")
    print("3. 简单测试播放器")
    print("4. 退出")

    choice = input("请选择: ").strip()

    if choice == "1":
        check_system_environment()
    elif choice == "2":
        debug_extracted_objects()
    elif choice == "3":
        create_simple_test_player()
    elif choice == "4":
        print("退出")
    else:
        print("无效选择")