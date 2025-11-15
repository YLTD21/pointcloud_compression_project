# scripts/step2_fast_high_value.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys
try:
    from gpu_utils import calculate_curvature_gpu, fast_voxel_downsample_gpu, GPUAccelerator
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def fast_extract_high_value_features_gpu(points, colors=None, labels=None,
                                        curvature_threshold=0.02,
                                        min_points_per_cluster=5,
                                        use_gpu=True):
    """
    GPU加速的高价值特征点提取
    """
    if len(points) < 20:
        return points, colors, labels

    # GPU加速曲率计算
    if use_gpu and HAS_GPU:
        print("  GPU加速曲率计算...", end="")
        curvatures = calculate_curvature_gpu(points, k=15)
        print("完成")
    else:
        print("  CPU曲率计算...", end="")
        from utils import calculate_point_curvature
        curvatures = calculate_point_curvature(points, k=15)
        print("完成")
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils import calculate_point_curvature, save_pointcloud, load_extracted_pointcloud


def fast_extract_high_value_features_gpu(points, colors=None, labels=None,
                                         curvature_threshold=0.02,
                                         min_points_per_cluster=5,
                                         use_gpu=True):
    """
    GPU加速的高价值特征点提取 - 修复版本
    """
    # 检查输入有效性
    if points is None or len(points) == 0:
        print("  警告: 输入点云为空")
        return np.array([]), colors, labels

    if len(points) < 20:
        print(f"  警告: 点云点数过少 ({len(points)})，跳过处理")
        return points, colors, labels

    try:
        # GPU加速曲率计算
        if use_gpu and HAS_GPU:
            print("  GPU加速曲率计算...", end="")
            curvatures = calculate_curvature_gpu(points, k=15)
            print("完成")
        else:
            print("  CPU曲率计算...", end="")
            from utils import calculate_point_curvature
            curvatures = calculate_point_curvature(points, k=15)
            print("完成")

        # 检查曲率计算结果
        if curvatures is None or len(curvatures) != len(points):
            print("  警告: 曲率计算失败，返回原始点云")
            return points, colors, labels

        # 基于曲率过滤
        if len(curvatures) > 0:
            threshold = np.percentile(curvatures, 70)
            high_curvature_mask = curvatures > threshold

            # 如果过滤后点数太少，调整阈值
            if np.sum(high_curvature_mask) < min_points_per_cluster * 3:
                threshold = np.percentile(curvatures, 50)
                high_curvature_mask = curvatures > threshold

            filtered_points = points[high_curvature_mask]
            filtered_colors = colors[high_curvature_mask] if colors is not None else None
            filtered_labels = labels[high_curvature_mask] if labels is not None else None

            print(f"  完成 - 原始点数: {len(points)}, 高价值点数: {len(filtered_points)}")

            return filtered_points, filtered_colors, filtered_labels
        else:
            print("  警告: 曲率数组为空，返回原始点云")
            return points, colors, labels

    except Exception as e:
        print(f"  高价值特征提取失败: {e}")
        print("  返回原始点云作为回退")
        return points, colors, labels


def process_fast_high_value_extraction(sequence_ids=None, max_files_per_sequence=100, use_gpu=True):
    """GPU加速的高价值特征点提取 - 修复版本"""
    if sequence_ids is None:
        sequence_ids = ["00"]

    processed_dir = project_root / "data" / "processed_dataset_final"
    high_value_dir = project_root / "data" / "high_value_dataset_fast"

    os.makedirs(high_value_dir, exist_ok=True)

    total_files_processed = 0
    total_points_original = 0
    total_points_high_value = 0

    for seq in sequence_ids:
        seq_input_dir = processed_dir / f"seq_{seq}"
        seq_output_dir = high_value_dir / f"seq_{seq}"
        os.makedirs(seq_output_dir, exist_ok=True)

        if not seq_input_dir.exists():
            print(f"序列 {seq} 不存在，跳过")
            continue

        # 获取PCD文件
        pcd_files = [f for f in seq_input_dir.iterdir() if f.is_file() and f.suffix == '.pcd']
        pcd_files.sort(key=lambda x: x.name)

        # 限制处理文件数量
        if max_files_per_sequence and len(pcd_files) > max_files_per_sequence:
            pcd_files = pcd_files[:max_files_per_sequence]
            print(f"序列 {seq}: 只处理前 {max_files_per_sequence} 个文件")

        print(f"\n处理序列 {seq}: {len(pcd_files)} 个文件")

        seq_points_original = 0
        seq_points_high_value = 0
        failed_files = 0

        for pcd_file in tqdm(pcd_files, desc=f"GPU加速提取序列 {seq}"):
            try:
                # 加载点云数据
                points, colors, labels = load_extracted_pointcloud(pcd_file)

                # 检查数据有效性
                if points is None or len(points) == 0:
                    print(f"\n  跳过空点云文件: {pcd_file.name}")
                    failed_files += 1
                    continue

                seq_points_original += len(points)

                # GPU加速提取高价值特征点
                result = fast_extract_high_value_features_gpu(
                    points, colors, labels, use_gpu=use_gpu
                )

                # 检查返回值
                if result is None:
                    print(f"\n  警告: {pcd_file.name} 返回None，跳过")
                    failed_files += 1
                    continue

                high_value_points, high_value_colors, high_value_labels = result

                # 检查提取结果
                if high_value_points is None or len(high_value_points) == 0:
                    print(f"\n  警告: {pcd_file.name} 未提取到高价值点")
                    # 仍然保存空结果以便统计
                    high_value_points = np.array([])
                    high_value_colors = None
                    high_value_labels = None

                seq_points_high_value += len(high_value_points)

                # 保存高价值点云
                output_filename = pcd_file.stem + '_high_value.pcd'
                output_file = seq_output_dir / output_filename

                # 只有有点时才保存PCD文件
                if len(high_value_points) > 0:
                    save_pointcloud(high_value_points, str(output_file), high_value_colors, high_value_labels)
                else:
                    # 对于空点云，创建一个空的PCD文件或跳过
                    print(f"  跳过保存空点云: {output_filename}")
                    continue

                # 保存统计信息（即使为空也保存）
                npy_filename = pcd_file.stem + '_high_value.npy'
                npy_output = seq_output_dir / npy_filename
                np.save(str(npy_output), {
                    'points': high_value_points,
                    'colors': high_value_colors,
                    'labels': high_value_labels,
                    'original_count': len(points),
                    'compressed_count': len(high_value_points),
                    'compression_ratio': len(high_value_points) / len(points) if len(points) > 0 else 0
                })

            except Exception as e:
                print(f"\n处理文件 {pcd_file.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                failed_files += 1
                continue

        total_files_processed += len(pcd_files)
        total_points_original += seq_points_original
        total_points_high_value += seq_points_high_value

        compression_ratio = seq_points_high_value / seq_points_original if seq_points_original > 0 else 0
        print(f"序列 {seq}: 压缩率 {compression_ratio:.2%}, 失败文件: {failed_files}")

    print(f"\n=== 快速提取完成 ===")
    print(f"处理了 {total_files_processed} 个文件")
    print(f"原始总点数: {total_points_original}")
    print(f"高价值总点数: {total_points_high_value}")
    print(f"总体压缩率: {total_points_high_value / total_points_original:.2%}")
    print(f"输出目录: {high_value_dir}")


def estimate_processing_time(sequence_ids=None):
    """估算处理时间"""
    if sequence_ids is None:
        sequence_ids = ["00"]

    processed_dir = project_root / "data" / "processed_dataset_final"

    total_files = 0
    for seq in sequence_ids:
        seq_dir = processed_dir / f"seq_{seq}"
        if seq_dir.exists():
            # 修复：使用Path对象的正确方法
            pcd_files = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix == '.pcd']
            total_files += len(pcd_files)

    estimated_time = total_files * 3
    minutes = estimated_time // 60
    seconds = estimated_time % 60

    print(f"估算信息:")
    print(f"  文件数量: {total_files}")
    print(f"  预计时间: {minutes}分{seconds}秒")
    print(f"  输出目录: data/high_value_dataset_fast/")


# 简单的测试函数
def test_single_file(sequence_id="00", file_index=0):
    """测试单个文件"""
    processed_dir = project_root / "data" / "processed_dataset_final"
    seq_dir = processed_dir / f"seq_{sequence_id}"

    if not seq_dir.exists():
        print(f"序列 {sequence_id} 不存在")
        return

    pcd_files = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix == '.pcd']
    pcd_files.sort(key=lambda x: x.name)

    if file_index >= len(pcd_files):
        print(f"文件索引 {file_index} 超出范围")
        return

    test_file = pcd_files[file_index]
    print(f"测试文件: {test_file.name}")

    try:
        points, colors, labels = load_extracted_pointcloud(test_file)
        print(f"原始点数: {len(points)}")

        high_value_points, high_value_colors, high_value_labels = fast_extract_high_value_features_gpu(
            points, colors, labels
        )
        print(f"高价值点数: {len(high_value_points)}")
        print(f"压缩率: {len(high_value_points) / len(points):.2%}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='快速高价值点云提取')
    parser.add_argument('--sequences', nargs='+', help='要处理的序列ID (例如: 00 01)')
    parser.add_argument('--max_files', type=int, default=100, help='每序列最大处理文件数')
    parser.add_argument('--estimate', action='store_true', help='只估算处理时间')
    parser.add_argument('--test', type=int, help='测试单个文件（文件索引）')

    args = parser.parse_args()

    sequences = args.sequences if args.sequences else ["00"]

    if args.test is not None:
        test_single_file("00", args.test)
    elif args.estimate:
        estimate_processing_time(sequences)
    else:
        print("开始快速高价值点云提取...")
        process_fast_high_value_extraction(sequences, args.max_files)