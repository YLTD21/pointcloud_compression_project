import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# 添加路径以确保模块导入
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils import (load_kitti_pointcloud, extract_pedestrian_vehicle_points,
                   save_pointcloud, find_semantic_kitti_sequences, get_project_root)


def process_semantic_kitti_data():
    """处理SemanticKITTI数据集，提取行人和车辆"""

    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed_dataset"

    # 创建输出目录
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 自动查找数据序列
    sequences = find_semantic_kitti_sequences()

    if not sequences:
        print("错误：未找到任何SemanticKITTI数据序列！")
        print("请检查数据目录结构，应该是：data/raw_dataset/dataset/sequences/00/velodyne/")
        return

    print(f"找到 {len(sequences)} 个数据序列")

    total_files_processed = 0
    total_points_extracted = 0

    for seq_info in sequences:
        seq_id = seq_info['seq_id']
        velodyne_path = seq_info['velodyne_path']
        labels_path = seq_info['labels_path']

        print(f"\n处理序列 {seq_id}:")
        print(f"  点云路径: {velodyne_path}")
        print(f"  标签路径: {labels_path}")

        # 创建序列输出目录
        seq_output_dir = processed_dir / f"seq_{seq_id}"
        seq_output_dir.mkdir(exist_ok=True)

        # 获取点云文件
        bin_files = list(velodyne_path.glob("*.bin"))

        if not bin_files:
            print(f"  警告：序列 {seq_id} 中没有找到.bin文件")
            continue

        print(f"  找到 {len(bin_files)} 个点云文件")

        seq_files_processed = 0
        seq_points_extracted = 0

        for bin_file in tqdm(bin_files, desc=f"处理序列 {seq_id}"):
            try:
                # 构建标签文件路径
                if labels_path:
                    label_file = bin_file.name.replace('.bin', '.label')
                    label_path_full = labels_path / label_file
                else:
                    label_path_full = None

                # 加载点云
                points, _ = load_kitti_pointcloud(str(bin_file))

                # 如果有标签文件，加载标签
                if label_path_full and label_path_full.exists():
                    labels = np.fromfile(label_path_full, dtype=np.uint32)
                    # SemanticKITTI标签是32位，取低16位是实例标签，高16位是语义标签
                    labels = labels & 0xFFFF  # 取语义标签
                else:
                    # 如果没有标签文件，创建一个全零的标签数组
                    labels = np.zeros(len(points), dtype=np.uint32)
                    print(f"  警告：{bin_file.name} 没有对应的标签文件")

                # 提取行人和车辆
                filtered_points, filtered_labels = extract_pedestrian_vehicle_points(points, labels)

                if len(filtered_points) > 0:
                    # 保存提取的点云
                    output_file = seq_output_dir / bin_file.name.replace('.bin', '.pcd')
                    save_pointcloud(filtered_points, str(output_file), filtered_labels)

                    # 同时保存npy格式用于后续处理
                    npy_file = seq_output_dir / bin_file.name.replace('.bin', '.npy')
                    np.save(str(npy_file), {
                        'points': filtered_points,
                        'labels': filtered_labels,
                        'source_file': str(bin_file)
                    })

                    seq_files_processed += 1
                    seq_points_extracted += len(filtered_points)

            except Exception as e:
                print(f"处理文件 {bin_file} 时出错: {e}")
                continue

        total_files_processed += seq_files_processed
        total_points_extracted += seq_points_extracted

        print(f"  序列 {seq_id} 完成: 处理了 {seq_files_processed} 个文件，提取了 {seq_points_extracted} 个点")

    print(f"\n=== 第一步完成 ===")
    print(f"总共处理了 {total_files_processed} 个文件")
    print(f"总共提取了 {total_points_extracted} 个行人和车辆点")
    print(f"输出目录: {processed_dir}")


# 独立的执行函数，供main.py调用
def execute_step1():
    """供main.py调用的执行函数"""
    print("开始执行第一步：提取行人和车辆数据")
    process_semantic_kitti_data()


if __name__ == "__main__":
    process_semantic_kitti_data()