import os
import numpy as np
import argparse
from tqdm import tqdm
import open3d as o3d

# --------------------------
# 核心配置（SemanticKITTI官方标签）
# --------------------------
SEMANTIC_MAP = {
    0: 0,  # 背景（不保留）
    1: 1,  # 汽车（保留，红色）
    4: 2  # 行人（保留，绿色）
}
TARGET_CLASSES = {1, 2}  # 只保留汽车和行人


# --------------------------
# 路径处理（适配你的数据集结构）
# --------------------------
def get_paths(raw_dir, seq):
    """
    你的路径结构：raw_dir/dataset/sequences/seq/velodyne & labels
    例如：data/raw_dataset/dataset/sequences/00/velodyne
    """
    # 拼接序列根目录（关键调整：增加dataset/sequences层级）
    seq_root = os.path.join(raw_dir, "dataset", "sequences", seq)
    velo_dir = os.path.join(seq_root, "velodyne")  # 点云文件目录
    label_dir = os.path.join(seq_root, "labels")  # 标签文件目录

    # 检查路径是否存在
    if not os.path.exists(velo_dir):
        raise FileNotFoundError(f"点云目录不存在：{velo_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"标签目录不存在：{label_dir}")

    # 获取所有帧文件（按序号排序）
    velo_files = sorted([f for f in os.listdir(velo_dir) if f.endswith(".bin")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".label")])

    # 确保点云和标签数量一致
    assert len(velo_files) == len(label_files), f"序列{seq}的点云和标签数量不匹配"
    return [(os.path.join(velo_dir, v), os.path.join(label_dir, l))
            for v, l in zip(velo_files, label_files)]


# --------------------------
# 点云处理（核心逻辑不变）
# --------------------------
def load_pointcloud(bin_path):
    """加载点云（x, y, z, intensity → 保留xyz）"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def load_semantic_labels(label_path):
    """提取纯语义ID（SemanticKITTI格式：右移16位）"""
    full_labels = np.fromfile(label_path, dtype=np.int32)
    return full_labels >> 16  # 过滤实例ID，保留语义ID


def filter_targets(points, semantic_ids):
    """过滤出汽车（1）和行人（4）"""
    mapped = np.zeros_like(semantic_ids, dtype=np.int32)
    for raw_id, target_id in SEMANTIC_MAP.items():
        mapped[semantic_ids == raw_id] = target_id
    mask = np.isin(mapped, list(TARGET_CLASSES))
    return points[mask], mapped[mask]


# --------------------------
# 保存与可视化
# --------------------------
def save_pcd(points, classes, save_path):
    """保存带颜色的点云（汽车红，行人绿）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((len(points), 3))
    colors[classes == 1] = [1, 0, 0]  # 汽车：红色
    colors[classes == 2] = [0, 1, 0]  # 行人：绿色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)


def visualize_result(points, classes, frame_id):
    """可视化单帧结果（调试用）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((len(points), 3))
    colors[classes == 1] = [1, 0, 0]
    colors[classes == 2] = [0, 1, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=f"帧 {frame_id}（红=汽车，绿=行人）")


# --------------------------
# 主函数
# --------------------------
def process_sequence(raw_dir, save_dir, seq, visualize=False):
    """处理单个序列"""
    # 创建保存目录（保持与原始结构对应）
    save_seq_dir = os.path.join(save_dir, "dataset", "sequences", seq, "pointcloud")
    os.makedirs(save_seq_dir, exist_ok=True)

    # 获取帧路径
    frame_paths = get_paths(raw_dir, seq)
    print(f"▶️ 开始处理序列{seq}（共{len(frame_paths)}帧）")

    # 逐帧处理
    for i, (velo_path, label_path) in enumerate(tqdm(frame_paths, desc=f"序列{seq}进度")):
        frame_id = os.path.basename(velo_path).split(".")[0]

        # 加载数据
        points = load_pointcloud(velo_path)
        semantic_ids = load_semantic_labels(label_path)

        # 过滤目标
        filtered_points, filtered_classes = filter_targets(points, semantic_ids)

        # 保存/可视化
        if len(filtered_points) == 0:
            print(f"ℹ️ 帧{frame_id}无汽车/行人")
        else:
            save_path = os.path.join(save_seq_dir, f"{frame_id}.pcd")
            save_pcd(filtered_points, filtered_classes, save_path)
            if visualize:
                visualize_result(filtered_points, filtered_classes, frame_id)

    print(f"✅ 序列{seq}处理完成，结果保存至 → {save_seq_dir}\n")


# --------------------------
# 入口
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取汽车和行人点云（适配你的路径）")
    parser.add_argument("--raw_dir", type=str,
                        default="/media/yml/share/pointcloud_compression_project/data/raw_dataset",
                        help="原始数据集根目录（包含dataset/sequences）")
    parser.add_argument("--save_dir", type=str,
                        default="/media/yml/share/pointcloud_compression_project/data/processed_dataset",
                        help="结果保存目录")
    parser.add_argument("--sequences", nargs="+", default=["00"],
                        help="处理的序列（如 00 01）")
    parser.add_argument("--visualize", action="store_true",
                        help="可视化每帧结果")
    args = parser.parse_args()

    for seq in args.sequences:
        process_sequence(args.raw_dir, args.save_dir, seq, args.visualize)
