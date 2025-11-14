# scripts/step1_extract_objects_final.py
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_kitti_data(bin_path, label_path):
    """加载KITTI点云和标签数据"""
    points_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = points_data[:, :3]

    if os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
        semantic_labels = labels & 0xFFFF  # 取低16位作为语义标签
        return points, semantic_labels
    else:
        return points, np.zeros(len(points), dtype=np.uint32)


def extract_objects_final(points, labels, background_radius=15.0, min_object_points=5):
    """
    基于发现的标签映射提取对象
    车辆: 1, 10, 51
    行人: 255
    """
    # 基于发现的实际标签映射
    vehicle_labels = [1, 10, 51]  # 车辆标签
    pedestrian_labels = [255]  # 行人标签

    vehicle_mask = np.isin(labels, vehicle_labels)
    pedestrian_mask = np.isin(labels, pedestrian_labels)
    object_mask = vehicle_mask | pedestrian_mask

    # 统计对象数量
    vehicle_count = np.sum(vehicle_mask)
    pedestrian_count = np.sum(pedestrian_mask)

    # 如果没有足够的对象点，返回空
    if np.sum(object_mask) < min_object_points:
        return np.array([]), np.array([]), vehicle_count, pedestrian_count

    # # 获取所有对象点的坐标
    # object_points = points[object_mask]
    # object_labels = labels[object_mask]
    #
    # # 计算对象点的边界框
    # min_bound = np.min(object_points, axis=0)
    # max_bound = np.max(object_points, axis=0)
    # center = (min_bound + max_bound) / 2
    #
    # # 扩展边界框以包含背景
    # expanded_min = center - background_radius
    # expanded_max = center + background_radius
    #
    # # 创建背景掩码（在扩展边界框内的非对象点）
    # background_mask = ~object_mask
    # for i in range(3):
    #     background_mask &= (points[:, i] >= expanded_min[i])
    #     background_mask &= (points[:, i] <= expanded_max[i])
    #
    # # 合并对象点和背景点
    # all_points = np.vstack([object_points, points[background_mask]])
    # all_labels = np.hstack([object_labels, labels[background_mask]])
    #
    # # 为点云创建颜色
    # colors = create_colors_final(all_labels)
    #
    # return all_points, colors, vehicle_count, pedestrian_count
    # 只保留对象点，去除背景
    object_points = points[object_mask]
    object_labels = labels[object_mask]

    # 为点云创建颜色
    colors = create_colors_final(object_labels)

    return object_points, colors, vehicle_count, pedestrian_count

def create_colors_final(labels):
    """
    基于最终确定的标签创建颜色
    车辆(1,10,51): 红色, 行人(255): 绿色, 背景: 灰色
    """
    colors = np.zeros((len(labels), 3))

    vehicle_labels = [1, 10, 51]
    pedestrian_labels = [255]

    for i, label in enumerate(labels):
        if label in vehicle_labels:  # 车辆 - 红色
            colors[i] = [1, 0, 0]
        elif label in pedestrian_labels:  # 行人 - 绿色
            colors[i] = [0, 1, 0]
        # else:  # 背景 - 灰色
        #     colors[i] = [0.4, 0.4, 0.4]  # 中等灰色

    return colors


def save_colored_pointcloud(points, colors, filename):
    """保存带颜色的点云"""
    if len(points) == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        o3d.io.write_point_cloud(filename, pcd)
        return True
    except Exception as e:
        print(f"保存点云失败 {filename}: {e}")
        return False


def process_semantic_kitti_data_final():
    """使用最终确定的标签映射处理所有数据"""

    raw_data_dir = project_root / "data" / "raw_dataset"
    processed_dir = project_root / "data" / "processed_dataset_final"

    processed_dir.mkdir(parents=True, exist_ok=True)

    sequences = [d for d in os.listdir(raw_data_dir / "dataset" / "sequences")
                 if os.path.isdir(raw_data_dir / "dataset" / "sequences" / d) and d.isdigit()]

    sequences.sort()

    total_frames_processed = 0
    total_frames_with_objects = 0
    total_vehicle_points = 0
    total_pedestrian_points = 0

    for seq in sequences:
        seq_path = raw_data_dir / "dataset" / "sequences" / seq / "velodyne"
        label_path = raw_data_dir / "dataset" / "sequences" / seq / "labels"

        if not seq_path.exists():
            print(f"序列 {seq} 的点云路径不存在: {seq_path}")
            continue

        seq_output_dir = processed_dir / f"seq_{seq}"
        seq_output_dir.mkdir(exist_ok=True)

        bin_files = [f for f in os.listdir(seq_path) if f.endswith('.bin')]
        bin_files.sort()

        seq_frames_with_objects = 0
        seq_vehicle_points = 0
        seq_pedestrian_points = 0

        print(f"\n处理序列 {seq}:")

        for bin_file in tqdm(bin_files, desc=f"序列 {seq}"):
            try:
                bin_file_path = seq_path / bin_file
                label_file = bin_file.replace('.bin', '.label')
                label_file_path = label_path / label_file

                # 加载数据
                points, labels = load_kitti_data(str(bin_file_path), str(label_file_path))

                # 使用最终确定的提取函数
                extracted_points, colors, vehicle_count, pedestrian_count = extract_objects_final(
                    points, labels, background_radius=15.0, min_object_points=10
                )

                seq_vehicle_points += vehicle_count
                seq_pedestrian_points += pedestrian_count

                if len(extracted_points) > 0:
                    output_file = seq_output_dir / bin_file.replace('.bin', '.pcd')
                    if save_colored_pointcloud(extracted_points, colors, str(output_file)):
                        seq_frames_with_objects += 1
                        total_frames_with_objects += 1

                    # 保存统计信息
                    npy_file = seq_output_dir / bin_file.replace('.bin', '.npy')
                    np.save(str(npy_file), {
                        'points': extracted_points,
                        'colors': colors,
                        'labels': labels,
                        'vehicle_count': vehicle_count,
                        'pedestrian_count': pedestrian_count,
                        'source_file': str(bin_file_path)
                    })

            except Exception as e:
                print(f"处理文件 {bin_file} 时出错: {e}")
                continue

        total_frames_processed += len(bin_files)
        total_vehicle_points += seq_vehicle_points
        total_pedestrian_points += seq_pedestrian_points

        print(f"序列 {seq}: 处理了 {len(bin_files)} 帧，其中 {seq_frames_with_objects} 帧包含对象")
        print(f"  车辆总点数: {seq_vehicle_points}, 行人总点数: {seq_pedestrian_points}")

    print(f"\n=== 处理完成 ===")
    print(f"总共处理了 {total_frames_processed} 帧")
    print(f"其中 {total_frames_with_objects} 帧包含对象")
    print(f"车辆总点数: {total_vehicle_points}")
    print(f"行人总点数: {total_pedestrian_points}")
    print(f"输出目录: {processed_dir}")


def test_final_extraction(sequence_id="00", file_indices=None):
    """测试最终提取方法"""
    if file_indices is None:
        file_indices = [0, 1, 2, 10, 20]

    raw_data_dir = project_root / "data" / "raw_dataset"
    seq_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "velodyne"
    label_path = raw_data_dir / "dataset" / "sequences" / sequence_id / "labels"

    bin_files = sorted(seq_path.glob("*.bin"))

    print(f"=== 测试最终提取方法 - 序列 {sequence_id} ===")

    for file_index in file_indices:
        if file_index >= len(bin_files):
            continue

        bin_file = bin_files[file_index]
        label_file = label_path / bin_file.name.replace('.bin', '.label')

        print(f"\n测试文件: {bin_file.name}")

        # 加载数据
        points, labels = load_kitti_data(str(bin_file), str(label_file))

        # 使用最终提取方法
        extracted_points, colors, vehicle_count, pedestrian_count = extract_objects_final(
            points, labels, background_radius=15.0, min_object_points=10
        )

        print(f"车辆点数: {vehicle_count}, 行人数: {pedestrian_count}")

        if len(extracted_points) > 0:
            print(f"提取点数: {len(extracted_points)}")

            # 可视化
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(extracted_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            print("显示提取结果...")
            o3d.visualization.draw_geometries([pcd],
                                              window_name=f"最终提取 - {bin_file.name}",
                                              width=1000, height=800)
        else:
            print("没有提取到对象")


def create_final_video_player():
    """为最终提取的数据创建专门的播放器"""
    # 这个函数可以调用之前创建的enhanced_objects_player.py
    # 但使用新的数据路径
    from enhanced_objects_player import EnhancedObjectsPlayer

    player = EnhancedObjectsPlayer()

    # 播放最终提取的序列
    for seq_id in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        success = player.play_enhanced_sequence(
            sequence_id=seq_id,
            frame_range=(0, 100),
            fps=5,
            point_size=4.0
        )

        if not success:
            print(f"序列 {seq_id} 播放失败或不存在")

        # 询问是否继续
        if seq_id != "10":
            response = input(f"继续播放下一个序列? (y/n): ")
            if response.lower() != 'y':
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='最终版点云对象提取')
    parser.add_argument('--process', action='store_true', help='处理所有数据')
    parser.add_argument('--test', type=str, help='测试提取方法 (例如: 00)')
    parser.add_argument('--play', action='store_true', help='播放最终提取的视频')

    args = parser.parse_args()

    if args.test:
        test_final_extraction(args.test)
    elif args.process:
        print("开始最终版点云对象提取...")
        process_semantic_kitti_data_final()
    elif args.play:
        create_final_video_player()
    else:
        print("请指定操作: --process, --test <序列ID>, 或 --play")

#该代码完成了我想要的行人与车辆提取，但是提取出来的点云数据有点大，但是效果不错，另外几个代码中的提取方法，车辆提取的不够完整，甚至没有提取出来，砸死使用debug_extracted_objects代码后，选择了一个合适的方法，提取了比较好的效果
#修改测试
#xiugaiceshi2
#现在我将原来生成的含有背景的点云放到了新加卷文件盘中，并且要改动该代码，使他不再提取背景类