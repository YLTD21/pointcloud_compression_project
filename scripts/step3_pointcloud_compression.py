import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import gzip
import pickle
import os
import sys

# 添加当前目录到Python路径，确保模块可以相互导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def voxel_downsample_compression(points, voxel_size=0.1):
    """体素下采样压缩"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points)


def octree_compression(points, compression_level=7):
    """八叉树压缩"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用Open3D的八叉树压缩
    compressed_data = o3d.io.write_point_cloud_to_octree(pcd, compression_level)
    return compressed_data


def statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0):
    """统计离群值去除"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)


def compress_and_save(points, output_path, method='voxel', **kwargs):
    """压缩并保存点云"""
    if method == 'voxel':
        compressed_points = voxel_downsample_compression(points, **kwargs)
    elif method == 'statistical':
        compressed_points = statistical_outlier_removal(points, **kwargs)
    else:
        compressed_points = points

    # 保存压缩后的点云
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(compressed_points)
    o3d.io.write_point_cloud(output_path, o3d_pcd)

    return compressed_points


def process_compression():
    """批量处理点云压缩"""

    high_value_dir = "../data/high_value_dataset"
    results_dir = "../results"

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
                original_count = data['original_count']
                high_value_count = len(points)

                for comp_method in compression_methods:
                    output_filename = npy_file.replace('.npy', f"_{comp_method['name']}.pcd")
                    output_path = os.path.join(results_dir, output_filename)

                    # 执行压缩
                    compressed_points = compress_and_save(points, output_path, **comp_method)

                    # 记录压缩结果
                    compression_ratio = len(compressed_points) / original_count
                    high_value_ratio = len(compressed_points) / high_value_count

                    compression_results.append({
                        'sequence': seq,
                        'file': npy_file,
                        'method': comp_method['name'],
                        'original_points': original_count,
                        'high_value_points': high_value_count,
                        'compressed_points': len(compressed_points),
                        'compression_ratio': compression_ratio,
                        'high_value_compression_ratio': high_value_ratio
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