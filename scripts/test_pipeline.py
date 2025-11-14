import numpy as np
import open3d as o3d
import os
from utils import *


def test_single_file():
    """测试单个文件处理流程"""

    # 创建测试数据
    test_points = np.random.rand(1000, 3) * 10
    test_labels = np.random.choice([252, 253, 1], 1000)  # 252:行人, 253:车辆

    print(f"原始点云数量: {len(test_points)}")

    # 步骤1: 提取行人和车辆
    filtered_points, filtered_labels = extract_pedestrian_vehicle_points(test_points, test_labels)
    print(f"提取后的点云数量: {len(filtered_points)}")

    # 步骤2: 提取高价值特征点
    from step2_extract_high_value_features import extract_high_value_features
    high_value_points, high_value_labels = extract_high_value_features(filtered_points, filtered_labels)
    print(f"高价值点云数量: {len(high_value_points)}")

    # 步骤3: 压缩
    from step3_pointcloud_compression import voxel_downsample_compression
    compressed_points = voxel_downsample_compression(high_value_points, voxel_size=0.5)
    print(f"压缩后点云数量: {len(compressed_points)}")

    # 保存测试结果
    test_dir = "../data/test_samples"
    os.makedirs(test_dir, exist_ok=True)

    save_pointcloud(test_points, os.path.join(test_dir, "original.pcd"), test_labels)
    save_pointcloud(filtered_points, os.path.join(test_dir, "filtered.pcd"), filtered_labels)
    save_pointcloud(high_value_points, os.path.join(test_dir, "high_value.pcd"), high_value_labels)
    save_pointcloud(compressed_points, os.path.join(test_dir, "compressed.pcd"))

    print("测试完成！结果保存在 data/test_samples/ 目录")


if __name__ == "__main__":
    test_single_file()