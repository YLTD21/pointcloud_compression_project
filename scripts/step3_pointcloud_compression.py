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
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_points = np.asarray(down_pcd.points)
    down_colors = np.asarray(down_pcd.colors) if down_pcd.has_colors() else None

    return down_points, down_colors


def octree_compression(points, colors=None, compression_level=7):
    """八叉树压缩 - 适配颜色信息"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    compressed_data = o3d.io.write_point_cloud_to_octree(pcd, compression_level)
    return compressed_data


def statistical_outlier_removal(points, colors=None, nb_neighbors=20, std_ratio=2.0):
    """统计离群值去除 - 适配颜色信息"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    cl_points = np.asarray(cl.points)
    cl_colors = np.asarray(cl.colors) if cl.has_colors() else None

    return cl_points, cl_colors


def compress_and_save(points, colors, output_path, method='voxel', **kwargs):
    """压缩并保存点云 - 适配新数据格式"""
    if method == 'voxel':
        compressed_points, compressed_colors = voxel_downsample_compression(points, colors, **kwargs)
    elif method == 'statistical':
        compressed_points, compressed_colors = statistical_outlier_removal(points, colors, **kwargs)
    else:
        compressed_points, compressed_colors = points, colors

    # 保存压缩后的点云
    save_pointcloud(compressed_points, output_path, compressed_colors)

    return compressed_points, compressed_colors


def process_compression():
    """批量处理点云压缩 - 适配新数据目录"""

    high_value_dir = project_root / "data" / "high_value_dataset"
    results_dir = project_root / "results"

    os.makedirs(results_dir, exist_ok=True)

    compression_methods = [
        {'name': 'voxel_0.1', 'method': 'voxel', 'voxel_size': 0.1},
        {'name': 'voxel_0.05', 'method': 'voxel', 'voxel_size': 0.05},
        {'name': 'statistical', 'method': 'statistical', 'nb_neighbors': 20, 'std_ratio': 2.0}
    ]

    sequences = [d for d in os.listdir(high_value_dir)
                 if os.path.isdir(os.path.join(high_value_dir, d))]

    compression_results = []

    for seq in sequences:
        seq_input_dir = os.path.join(high_value_dir, seq)

        npy_files = [f for f in os.listdir(seq_input_dir) if f.endswith('_high_value.npy')]

        for npy_file in tqdm(npy_files, desc=f"Compressing {seq}"):
            try:
                data = np.load(os.path.join(seq_input_dir, npy_file), allow_pickle=True).item()
                points = data['points']
                colors = data.get('colors', None)
                original_count = data['original_count']
                high_value_count = len(points)

                for comp_method in compression_methods:
                    output_filename = npy_file.replace('.npy', f"_{comp_method['name']}.pcd")
                    output_path = os.path.join(results_dir, output_filename)

                    # 执行压缩
                    compressed_points, compressed_colors = compress_and_save(
                        points, colors, output_path, **comp_method
                    )

                    # 记录压缩结果
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

            except Exception as e:
                print(f"Error compressing {npy_file}: {e}")
                continue

    # 保存压缩结果统计
    import pandas as pd
    df = pd.DataFrame(compression_results)
    df.to_csv(os.path.join(results_dir, 'compression_statistics.csv'), index=False)

    # 打印统计信息
    print("\n=== 压缩结果统计 ===")
    print(df.groupby('method')[['compression_ratio', 'high_value_compression_ratio']].mean())


if __name__ == "__main__":
    process_compression()