# scripts/step3_pointcloud_compression.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import gzip
import pickle
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils import save_pointcloud, load_extracted_pointcloud


def voxel_downsample_compression(points, colors=None, voxel_size=0.1):
    """体素下采样压缩 - 适配颜色信息"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if down_pcd is None or len(down_pcd.points) == 0:
        return points, colors  # 如果下采样后为空，返回原始点云

    down_points = np.asarray(down_pcd.points)
    down_colors = np.asarray(down_pcd.colors) if down_pcd.has_colors() else None

    return down_points, down_colors


def octree_compression(points, colors=None, compression_level=7):
    """八叉树压缩 - 适配颜色信息"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    try:
        compressed_data = o3d.io.write_point_cloud_to_octree(pcd, compression_level)
        return compressed_data
    except:
        # 如果八叉树压缩失败，返回原始数据
        return None


def statistical_outlier_removal(points, colors=None, nb_neighbors=20, std_ratio=2.0):
    """统计离群值去除 - 适配颜色信息"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    try:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cl_points = np.asarray(cl.points)
        cl_colors = np.asarray(cl.colors) if cl.has_colors() else None
        return cl_points, cl_colors
    except:
        # 如果离群值去除失败，返回原始数据
        return points, colors


def compress_and_save(points, colors, output_path, method='voxel', **kwargs):
    """压缩并保存点云 - 修复参数传递问题"""
    try:
        if method == 'voxel':
            # 只传递voxel_size参数
            voxel_size = kwargs.get('voxel_size', 0.1)
            compressed_points, compressed_colors = voxel_downsample_compression(
                points, colors, voxel_size=voxel_size
            )
        elif method == 'statistical':
            # 只传递统计去除相关参数
            nb_neighbors = kwargs.get('nb_neighbors', 20)
            std_ratio = kwargs.get('std_ratio', 2.0)
            compressed_points, compressed_colors = statistical_outlier_removal(
                points, colors, nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
        else:
            # 默认不压缩
            compressed_points, compressed_colors = points, colors

        # 保存压缩后的点云
        if len(compressed_points) > 0:
            save_pointcloud(compressed_points, output_path, compressed_colors)
            return compressed_points, compressed_colors
        else:
            print(f"  警告: 压缩后点云为空，跳过保存")
            return None, None

    except Exception as e:
        print(f"  压缩失败: {e}")
        return None, None


def process_compression():
    """批量处理点云压缩 - 修复版本"""

    # 检查输入目录是否存在
    high_value_dir = project_root / "data" / "high_value_dataset"
    high_value_fast_dir = project_root / "data" / "high_value_dataset_fast"

    # 优先使用标准高价值数据，如果不存在则使用快速版本
    if high_value_dir.exists():
        input_dir = high_value_dir
        print("使用标准高价值数据")
    elif high_value_fast_dir.exists():
        input_dir = high_value_fast_dir
        print("使用快速高价值数据")
    else:
        print("错误: 未找到高价值数据目录")
        print(f"请检查: {high_value_dir} 或 {high_value_fast_dir}")
        return

    results_dir = project_root / "results"
    os.makedirs(results_dir, exist_ok=True)

    # 修复压缩方法定义 - 只包含必要的参数
    compression_methods = [
        {'name': 'voxel_0.1', 'method': 'voxel', 'voxel_size': 0.1},
        {'name': 'voxel_0.05', 'method': 'voxel', 'voxel_size': 0.05},
        {'name': 'voxel_0.2', 'method': 'voxel', 'voxel_size': 0.2},
        {'name': 'statistical_20_2.0', 'method': 'statistical', 'nb_neighbors': 20, 'std_ratio': 2.0},
        {'name': 'statistical_10_1.5', 'method': 'statistical', 'nb_neighbors': 10, 'std_ratio': 1.5}
    ]

    # 查找所有序列
    sequences = [d for d in os.listdir(input_dir)
                 if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('seq_')]

    if not sequences:
        print(f"在 {input_dir} 中未找到任何序列")
        return

    compression_results = []

    for seq in sequences:
        seq_input_dir = os.path.join(input_dir, seq)

        # 查找高价值npy文件
        npy_files = [f for f in os.listdir(seq_input_dir) if f.endswith('_high_value.npy')]

        if not npy_files:
            print(f"序列 {seq} 中没有高价值npy文件")
            continue

        print(f"\n处理序列 {seq}: {len(npy_files)} 个文件")

        for npy_file in tqdm(npy_files, desc=f"压缩 {seq}"):
            try:
                # 加载高价值数据
                data_path = os.path.join(seq_input_dir, npy_file)
                data = np.load(data_path, allow_pickle=True).item()

                points = data['points']
                colors = data.get('colors', None)
                original_count = data.get('original_count', len(points))
                high_value_count = len(points)

                # 检查点云是否为空
                if len(points) == 0:
                    print(f"  跳过空点云: {npy_file}")
                    continue

                for comp_method in compression_methods:
                    output_filename = npy_file.replace('.npy', f"_{comp_method['name']}.pcd")
                    output_path = os.path.join(results_dir, output_filename)

                    # 执行压缩 - 只传递必要的参数
                    method_name = comp_method['method']
                    method_params = {k: v for k, v in comp_method.items() if k not in ['name', 'method']}

                    compressed_points, compressed_colors = compress_and_save(
                        points, colors, output_path, method=method_name, **method_params
                    )

                    # 如果压缩成功，记录结果
                    if compressed_points is not None and len(compressed_points) > 0:
                        compression_ratio = len(compressed_points) / original_count
                        high_value_compression_ratio = len(compressed_points) / high_value_count

                        compression_results.append({
                            'sequence': seq,
                            'file': npy_file,
                            'method': comp_method['name'],
                            'original_points': original_count,
                            'high_value_points': high_value_count,
                            'compressed_points': len(compressed_points),
                            'compression_ratio': compression_ratio,
                            'high_value_compression_ratio': high_value_compression_ratio
                        })
                    else:
                        print(f"  压缩失败或结果为空: {npy_file} - {comp_method['name']}")

            except Exception as e:
                print(f"处理文件 {npy_file} 时出错: {e}")
                continue

    # 保存压缩结果统计
    if compression_results:
        import pandas as pd
        df = pd.DataFrame(compression_results)
        stats_file = os.path.join(results_dir, 'compression_statistics.csv')
        df.to_csv(stats_file, index=False)

        # 打印统计信息
        print("\n=== 压缩结果统计 ===")
        print(f"总共处理了 {len(compression_results)} 个压缩结果")

        try:
            stats = df.groupby('method')[['compression_ratio', 'high_value_compression_ratio']].mean()
            print(stats)

            # 保存详细统计
            summary_file = os.path.join(results_dir, 'compression_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("点云压缩统计摘要\n")
                f.write("=" * 50 + "\n")
                f.write(f"总压缩结果数: {len(compression_results)}\n")
                f.write(f"压缩方法: {df['method'].unique().tolist()}\n\n")
                f.write("各方法平均压缩率:\n")
                f.write(stats.to_string())

            print(f"\n详细统计已保存到: {stats_file} 和 {summary_file}")

        except Exception as e:
            print(f"生成统计信息时出错: {e}")
            print("压缩结果数据:")
            print(df.head())
    else:
        print("\n警告: 没有成功的压缩结果")
        print("请检查高价值数据文件和压缩参数")


def test_single_compression(sequence_id="00", file_index=0, method_name="voxel_0.1"):
    """测试单个文件的压缩"""
    # 确定输入目录
    high_value_dir = project_root / "data" / "high_value_dataset"
    high_value_fast_dir = project_root / "data" / "high_value_dataset_fast"

    if high_value_dir.exists():
        input_dir = high_value_dir
    elif high_value_fast_dir.exists():
        input_dir = high_value_fast_dir
    else:
        print("错误: 未找到高价值数据目录")
        return

    seq_dir = input_dir / f"seq_{sequence_id}"

    if not seq_dir.exists():
        print(f"序列 {sequence_id} 不存在")
        return

    npy_files = [f for f in os.listdir(seq_dir) if f.endswith('_high_value.npy')]
    npy_files.sort()

    if file_index >= len(npy_files):
        print(f"文件索引 {file_index} 超出范围")
        return

    npy_file = npy_files[file_index]
    print(f"测试压缩文件: {npy_file}")

    # 加载数据
    data_path = os.path.join(seq_dir, npy_file)
    data = np.load(data_path, allow_pickle=True).item()

    points = data['points']
    colors = data.get('colors', None)

    print(f"原始点数: {len(points)}")

    # 定义压缩方法
    compression_methods = {
        'voxel_0.1': {'method': 'voxel', 'voxel_size': 0.1},
        'voxel_0.05': {'method': 'voxel', 'voxel_size': 0.05},
        'statistical_20_2.0': {'method': 'statistical', 'nb_neighbors': 20, 'std_ratio': 2.0}
    }

    if method_name in compression_methods:
        method_config = compression_methods[method_name]
        compressed_points, compressed_colors = compress_and_save(
            points, colors, "test_compression.pcd", **method_config
        )

        if compressed_points is not None:
            print(f"压缩后点数: {len(compressed_points)}")
            print(f"压缩率: {len(compressed_points) / len(points):.2%}")

            # 可视化结果
            import open3d as o3d
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                original_pcd.colors = o3d.utility.Vector3dVector(colors)

            compressed_pcd = o3d.geometry.PointCloud()
            compressed_pcd.points = o3d.utility.Vector3dVector(compressed_points)
            if compressed_colors is not None:
                compressed_pcd.colors = o3d.utility.Vector3dVector(compressed_colors)

            print("显示原始点云...")
            o3d.visualization.draw_geometries([original_pcd], window_name="原始点云")

            print("显示压缩点云...")
            o3d.visualization.draw_geometries([compressed_pcd], window_name="压缩点云")
        else:
            print("压缩失败")
    else:
        print(f"未知的压缩方法: {method_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='点云压缩')
    parser.add_argument('--test', action='store_true', help='测试单个文件压缩')
    parser.add_argument('--sequence', default="00", help='测试序列ID')
    parser.add_argument('--file_index', type=int, default=0, help='测试文件索引')
    parser.add_argument('--method', default="voxel_0.1", help='压缩方法')

    args = parser.parse_args()

    if args.test:
        test_single_compression(args.sequence, args.file_index, args.method)
    else:
        process_compression()