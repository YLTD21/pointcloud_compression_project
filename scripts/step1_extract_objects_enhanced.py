# scripts/step1_extract_objects_enhanced.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_kitti_pointcloud(bin_file_path):
    """
    加载SemanticKITTI的bin文件
    返回: points (N,3), labels (N,)
    """
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], points[:, 3].astype(np.uint32)


def extract_objects_with_background(points, labels, background_radius=10.0):
    """
    提取行人和车辆，并保留周围背景
    background_radius: 围绕行人和车辆提取背景的半径（米）
    """
    # 找到行人和车辆的掩码
    pedestrian_mask = (labels == 252)
    vehicle_mask = (labels == 253)
    object_mask = pedestrian_mask | vehicle_mask

    # 如果没有检测到任何对象，返回空数组
    if not np.any(object_mask):
        return np.array([]), np.array([])

    # 获取所有对象点的坐标
    object_points = points[object_mask]
    object_labels = labels[object_mask]

    # 计算对象点的边界框
    if len(object_points) > 0:
        min_bound = np.min(object_points, axis=0)
        max_bound = np.max(object_points, axis=0)
        center = (min_bound + max_bound) / 2

        # 扩展边界框以包含背景
        expanded_min = center - background_radius
        expanded_max = center + background_radius

        # 创建背景掩码（在扩展边界框内的非对象点）
        background_mask = ~object_mask
        for i in range(3):
            background_mask &= (points[:, i] >= expanded_min[i])
            background_mask &= (points[:, i] <= expanded_max[i])

        # 合并对象点和背景点
        all_points = np.vstack([points[object_mask], points[background_mask]])
        all_labels = np.hstack([labels[object_mask], labels[background_mask]])

        # 为点云创建颜色
        colors = create_colors_based_on_labels(all_labels)

        return all_points, colors
    else:
        return np.array([]), np.array([])


def create_colors_based_on_labels(labels):
    """
    根据标签创建颜色
    行人(252): 绿色, 车辆(253): 红色, 背景: 灰色
    """
    colors = np.zeros((len(labels), 3))

    for i, label in enumerate(labels):
        if label == 252:  # 行人 - 绿色
            colors[i] = [0, 1, 0]
        elif label == 253:  # 车辆 - 红色
            colors[i] = [1, 0, 0]
        else:  # 背景 - 灰色
            colors[i] = [0.5, 0.5, 0.5]

    return colors


def save_colored_pointcloud(points, colors, filename):
    """保存带颜色的点云"""
    if len(points) == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        o3d.io.write_point_cloud(filename, pcd)
        return True
    except Exception as e:
        print(f"保存点云失败 {filename}: {e}")
        return False


def process_semantic_kitti_data_enhanced():
    """处理SemanticKITTI数据集，提取行人和车辆及周围背景"""

    raw_data_dir = project_root / "data" / "raw_dataset"
    processed_dir = project_root / "data" / "processed_dataset_enhanced"

    # 创建输出目录
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有序列文件夹
    sequences = [d for d in os.listdir(raw_data_dir / "dataset" / "sequences")
                 if os.path.isdir(raw_data_dir / "dataset" / "sequences" / d) and d.isdigit()]

    sequences.sort()  # 按顺序处理

    total_frames_processed = 0
    total_frames_with_objects = 0

    for seq in sequences:
        seq_path = raw_data_dir / "dataset" / "sequences" / seq / "velodyne"
        label_path = raw_data_dir / "dataset" / "sequences" / seq / "labels"

        if not seq_path.exists():
            print(f"序列 {seq} 的点云路径不存在: {seq_path}")
            continue

        # 创建序列输出目录
        seq_output_dir = processed_dir / f"seq_{seq}"
        seq_output_dir.mkdir(exist_ok=True)

        # 处理每个点云文件
        bin_files = [f for f in os.listdir(seq_path) if f.endswith('.bin')]
        bin_files.sort()  # 按文件名排序

        seq_frames_with_objects = 0

        for bin_file in tqdm(bin_files, desc=f"处理序列 {seq}"):
            try:
                bin_path = seq_path / bin_file
                label_file = bin_file.replace('.bin', '.label')
                label_path_full = label_path / label_file

                # 加载点云和标签
                points, labels = load_kitti_pointcloud(str(bin_path))

                # 提取对象和周围背景
                extracted_points, colors = extract_objects_with_background(points, labels, background_radius=15.0)

                # 如果有对象，保存点云
                if len(extracted_points) > 0:
                    # 保存PCD文件
                    output_file = seq_output_dir / bin_file.replace('.bin', '.pcd')
                    if save_colored_pointcloud(extracted_points, colors, str(output_file)):
                        seq_frames_with_objects += 1
                        total_frames_with_objects += 1

                    # 保存NPY格式用于后续处理
                    npy_file = seq_output_dir / bin_file.replace('.bin', '.npy')
                    np.save(str(npy_file), {
                        'points': extracted_points,
                        'colors': colors,
                        'labels': labels[np.where((labels == 252) | (labels == 253))[0]] if np.any(
                            (labels == 252) | (labels == 253)) else np.array([]),
                        'source_file': str(bin_path)
                    })

            except Exception as e:
                print(f"处理文件 {bin_file} 时出错: {e}")
                continue

        total_frames_processed += len(bin_files)
        print(f"序列 {seq}: 处理了 {len(bin_files)} 帧，其中 {seq_frames_with_objects} 帧包含行人和车辆")

    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {total_frames_processed} 帧")
    print(f"其中 {total_frames_with_objects} 帧包含行人和车辆")
    print(f"输出目录: {processed_dir}")


def analyze_extraction_results():
    """分析提取结果"""
    processed_dir = project_root / "data" / "processed_dataset_enhanced"

    if not processed_dir.exists():
        print("增强版处理结果目录不存在，请先运行处理脚本")
        return

    sequences = [d for d in os.listdir(processed_dir)
                 if os.path.isdir(processed_dir / d) and d.startswith('seq_')]

    print("=== 增强版提取结果分析 ===")

    for seq in sequences:
        seq_dir = processed_dir / seq
        pcd_files = list(seq_dir.glob("*.pcd"))

        if not pcd_files:
            continue

        # 统计点云信息
        point_counts = []
        for pcd_file in pcd_files[:20]:  # 只检查前20个文件
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                # 统计各类点数
                if len(points) > 0 and len(colors) > 0:
                    # 根据颜色判断类别
                    red_points = np.sum(np.all(np.isclose(colors, [1, 0, 0]), axis=1))  # 车辆
                    green_points = np.sum(np.all(np.isclose(colors, [0, 1, 0]), axis=1))  # 行人
                    gray_points = np.sum(np.all(np.isclose(colors, [0.5, 0.5, 0.5]), axis=1))  # 背景

                    point_counts.append({
                        'total': len(points),
                        'vehicles': red_points,
                        'pedestrians': green_points,
                        'background': gray_points
                    })
            except Exception as e:
                print(f"分析文件 {pcd_file} 时出错: {e}")

        if point_counts:
            avg_total = np.mean([pc['total'] for pc in point_counts])
            avg_vehicles = np.mean([pc['vehicles'] for pc in point_counts])
            avg_pedestrians = np.mean([pc['pedestrians'] for pc in point_counts])
            avg_background = np.mean([pc['background'] for pc in point_counts])

            print(f"序列 {seq}:")
            print(f"  平均总点数: {avg_total:.1f}")
            print(f"  平均车辆点数: {avg_vehicles:.1f}")
            print(f"  平均行人数: {avg_pedestrians:.1f}")
            print(f"  平均背景点数: {avg_background:.1f}")
            print(f"  对象占比: {(avg_vehicles + avg_pedestrians) / avg_total * 100:.1f}%")


def visualize_sample_frames(sequence_id="00", num_frames=5):
    """可视化样本帧以检查效果"""
    processed_dir = project_root / "data" / "processed_dataset_enhanced"
    seq_dir = processed_dir / f"seq_{sequence_id}"

    if not seq_dir.exists():
        print(f"序列 {sequence_id} 的增强版处理结果不存在")
        return

    pcd_files = sorted(seq_dir.glob("*.pcd"))

    if not pcd_files:
        print(f"序列 {sequence_id} 没有点云文件")
        return

    print(f"可视化序列 {sequence_id} 的样本帧...")

    for i, pcd_file in enumerate(pcd_files[:num_frames]):
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            if len(points) == 0:
                print(f"帧 {pcd_file.name}: 空点云")
                continue

            # 统计各类点数
            red_points = np.sum(np.all(np.isclose(colors, [1, 0, 0], atol=0.1), axis=1))  # 车辆
            green_points = np.sum(np.all(np.isclose(colors, [0, 1, 0], atol=0.1), axis=1))  # 行人
            gray_points = np.sum(np.all(np.isclose(colors, [0.5, 0.5, 0.5], atol=0.1), axis=1))  # 背景

            print(f"帧 {pcd_file.name}:")
            print(f"  总点数: {len(points)}")
            print(f"  车辆点数: {red_points}")
            print(f"  行人数: {green_points}")
            print(f"  背景点数: {gray_points}")

            # 可视化
            o3d.visualization.draw_geometries([pcd],
                                              window_name=f"序列 {sequence_id} - 帧 {i + 1}",
                                              width=1000,
                                              height=800)

        except Exception as e:
            print(f"可视化帧 {pcd_file} 时出错: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='增强版点云对象提取')
    parser.add_argument('--process', action='store_true', help='处理数据')
    parser.add_argument('--analyze', action='store_true', help='分析结果')
    parser.add_argument('--visualize', type=str, help='可视化序列 (例如: 00)')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')

    args = parser.parse_args()

    if args.process or args.all:
        print("开始增强版点云对象提取...")
        process_semantic_kitti_data_enhanced()

    if args.analyze or args.all:
        analyze_extraction_results()

    if args.visualize:
        visualize_sample_frames(args.visualize)

    if not any(vars(args).values()):
        print("请指定操作: --process, --analyze, --visualize <序列ID>, 或 --all")