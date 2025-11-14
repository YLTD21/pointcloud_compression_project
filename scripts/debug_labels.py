# scripts/debug_labels.py
import numpy as np
from pathlib import Path
import open3d as o3d


def debug_label_loading(sequence_id="00", num_files=5):
    """调试标签加载过程"""
    print(f"=== 调试序列 {sequence_id} 的标签加载 ===")

    raw_dir = Path("../data/raw_dataset/dataset/sequences") / sequence_id
    velodyne_dir = raw_dir / "velodyne"
    labels_dir = raw_dir / "labels"

    bin_files = sorted(velodyne_dir.glob("*.bin"))[:num_files]

    for bin_file in bin_files:
        label_file = labels_dir / bin_file.name.replace('.bin', '.label')

        print(f"\n文件: {bin_file.name}")

        # 加载点云
        points_data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        points = points_data[:, :3]
        print(f"  点云点数: {len(points)}")

        # 加载标签
        if label_file.exists():
            labels = np.fromfile(label_file, dtype=np.uint32)
            print(f"  标签数量: {len(labels)}")

            # 检查标签值
            unique_labels = np.unique(labels)
            print(f"  唯一标签值: {unique_labels}")

            # 检查行人和车辆标签
            pedestrian_count = np.sum(labels == 252)
            vehicle_count = np.sum(labels == 253)
            print(f"  行人标签(252)数量: {pedestrian_count}")
            print(f"  车辆标签(253)数量: {vehicle_count}")

            # 检查标签的语义部分（低16位）
            semantic_labels = labels & 0xFFFF  # 取低16位
            unique_semantic = np.unique(semantic_labels)
            print(f"  语义标签(低16位): {unique_semantic}")

            pedestrian_semantic = np.sum(semantic_labels == 252)
            vehicle_semantic = np.sum(semantic_labels == 253)
            print(f"  语义标签-行人(252): {pedestrian_semantic}")
            print(f"  语义标签-车辆(253): {vehicle_semantic}")

        else:
            print(f"  标签文件不存在: {label_file}")


def check_label_mapping():
    """检查SemanticKITTI标签映射"""
    print("\n=== SemanticKITTI标签映射 ===")
    labels_mapping = {
        0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle',
        4: 'truck', 5: 'other-vehicle', 6: 'person', 7: 'bicyclist',
        8: 'motorcyclist', 9: 'road', 10: 'parking', 11: 'sidewalk',
        12: 'other-ground', 13: 'building', 14: 'fence', 15: 'vegetation',
        16: 'trunk', 17: 'terrain', 18: 'pole', 19: 'traffic-sign'
    }

    for label_id, label_name in labels_mapping.items():
        print(f"  {label_id}: {label_name}")


if __name__ == "__main__":
    debug_label_loading("00", 10)
    check_label_mapping()# scripts/debug_labels.py
import numpy as np
from pathlib import Path
import open3d as o3d

def debug_label_loading(sequence_id="00", num_files=5):
    """调试标签加载过程"""
    print(f"=== 调试序列 {sequence_id} 的标签加载 ===")

    raw_dir = Path("../data/raw_dataset/dataset/sequences") / sequence_id
    velodyne_dir = raw_dir / "velodyne"
    labels_dir = raw_dir / "labels"

    bin_files = sorted(velodyne_dir.glob("*.bin"))[:num_files]

    for bin_file in bin_files:
        label_file = labels_dir / bin_file.name.replace('.bin', '.label')

        print(f"\n文件: {bin_file.name}")

        # 加载点云
        points_data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        points = points_data[:, :3]
        print(f"  点云点数: {len(points)}")

        # 加载标签
        if label_file.exists():
            labels = np.fromfile(label_file, dtype=np.uint32)
            print(f"  标签数量: {len(labels)}")

            # 检查标签值
            unique_labels = np.unique(labels)
            print(f"  唯一标签值: {unique_labels}")

            # 检查行人和车辆标签
            pedestrian_count = np.sum(labels == 252)
            vehicle_count = np.sum(labels == 253)
            print(f"  行人标签(252)数量: {pedestrian_count}")
            print(f"  车辆标签(253)数量: {vehicle_count}")

            # 检查标签的语义部分（低16位）
            semantic_labels = labels & 0xFFFF  # 取低16位
            unique_semantic = np.unique(semantic_labels)
            print(f"  语义标签(低16位): {unique_semantic}")

            pedestrian_semantic = np.sum(semantic_labels == 252)
            vehicle_semantic = np.sum(semantic_labels == 253)
            print(f"  语义标签-行人(252): {pedestrian_semantic}")
            print(f"  语义标签-车辆(253): {vehicle_semantic}")

        else:
            print(f"  标签文件不存在: {label_file}")

def check_label_mapping():
    """检查SemanticKITTI标签映射"""
    print("\n=== SemanticKITTI标签映射 ===")
    labels_mapping = {
        0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle',
        4: 'truck', 5: 'other-vehicle', 6: 'person', 7: 'bicyclist',
        8: 'motorcyclist', 9: 'road', 10: 'parking', 11: 'sidewalk',
        12: 'other-ground', 13: 'building', 14: 'fence', 15: 'vegetation',
        16: 'trunk', 17: 'terrain', 18: 'pole', 19: 'traffic-sign'
    }

    for label_id, label_name in labels_mapping.items():
        print(f"  {label_id}: {label_name}")

if __name__ == "__main__":
    debug_label_loading("00", 10)
    check_label_mapping()