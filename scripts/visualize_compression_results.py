# scripts/visualize_compression_results.py
import open3d as o3d
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def visualize_compression_comparison(sequence_id="00", file_index=0):
    """对比显示原始高价值点云和不同压缩方法的结果"""

    # 原始高价值数据路径
    high_value_dir = project_root / "data" / "high_value_dataset_fast" / f"seq_{sequence_id}"
    results_dir = project_root / "results"

    if not high_value_dir.exists():
        print(f"高价值序列 {sequence_id} 不存在")
        return

    # 获取原始高价值文件
    high_value_files = [f for f in high_value_dir.iterdir()
                        if f.is_file() and f.suffix == '.pcd' and '_high_value' in f.stem]
    high_value_files.sort(key=lambda x: x.name)

    if file_index >= len(high_value_files):
        print(f"文件索引 {file_index} 超出范围")
        return

    original_file = high_value_files[file_index]
    print(f"原始文件: {original_file.name}")

    # 加载原始高价值点云
    original_pcd = o3d.io.read_point_cloud(str(original_file))
    original_points = np.asarray(original_pcd.points)

    if len(original_points) == 0:
        print("原始点云为空")
        return

    print(f"原始高价值点数: {len(original_points)}")

    # 查找对应的压缩文件
    base_name = original_file.stem
    compression_methods = [
        'voxel_0.05', 'voxel_0.1', 'voxel_0.2',
        'statistical_10_1.5', 'statistical_20_2.0'
    ]

    compressed_pcds = []
    pcd_names = ["原始高价值点云"]

    for method in compression_methods:
        compressed_file = results_dir / f"{base_name}_{method}.pcd"
        if compressed_file.exists():
            pcd = o3d.io.read_point_cloud(str(compressed_file))
            if len(pcd.points) > 0:
                compressed_pcds.append(pcd)
                pcd_names.append(f"{method} ({len(pcd.points)}点)")
                print(f"{method}: {len(pcd.points)} 点")

    if not compressed_pcds:
        print("未找到压缩结果文件")
        return

    # 为每个点云设置不同颜色以便区分
    colors = [
        [1, 1, 1],  # 原始 - 白色
        [1, 0, 0],  # voxel_0.05 - 红色
        [0, 1, 0],  # voxel_0.1 - 绿色
        [0, 0, 1],  # voxel_0.2 - 蓝色
        [1, 1, 0],  # statistical_10_1.5 - 黄色
        [1, 0, 1]  # statistical_20_2.0 - 紫色
    ]

    # 创建可视化几何体列表
    geometries = []

    # 添加原始点云
    original_pcd.paint_uniform_color(colors[0])
    geometries.append(original_pcd)

    # 添加压缩点云
    for i, pcd in enumerate(compressed_pcds):
        color_idx = min(i + 1, len(colors) - 1)
        pcd.paint_uniform_color(colors[color_idx])
        geometries.append(pcd)

    # 创建图例
    legend_geometries = create_legend(pcd_names, colors[:len(pcd_names)])
    geometries.extend(legend_geometries)

    # 可视化
    print("\n显示压缩对比结果:")
    print("白色: 原始 | 红色: voxel_0.05 | 绿色: voxel_0.1 | 蓝色: voxel_0.2")
    print("黄色: statistical_10_1.5 | 紫色: statistical_20_2.0")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"压缩效果对比 - {original_file.name}",
        width=1200,
        height=800
    )


def create_legend(names, colors):
    """创建图例"""
    legend_geometries = []

    for i, (name, color) in enumerate(zip(names, colors)):
        # 创建图例文本点云
        text = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        text.translate([i * 0.3, 0, 0])
        text.paint_uniform_color(color)
        legend_geometries.append(text)

    return legend_geometries


def show_compression_statistics():
    """显示压缩统计信息"""
    results_dir = project_root / "results"
    stats_file = results_dir / "compression_statistics.csv"

    if not stats_file.exists():
        print("压缩统计文件不存在")
        return

    df = pd.read_csv(stats_file)

    print("\n=== 压缩统计详情 ===")
    print(f"总压缩结果数: {len(df)}")

    # 按方法分组统计
    stats = df.groupby('method').agg({
        'original_points': 'mean',
        'high_value_points': 'mean',
        'compressed_points': 'mean',
        'compression_ratio': ['mean', 'std'],
        'high_value_compression_ratio': ['mean', 'std']
    }).round(4)

    print("\n各方法详细统计:")
    print(stats)

    # 最佳压缩方法推荐
    print("\n=== 推荐 ===")
    best_preservation = df.loc[df.groupby('method')['high_value_compression_ratio'].idxmax()]
    best_compression = df.loc[df.groupby('method')['compression_ratio'].idxmin()]

    print("最佳特征保留方法 (高价值压缩率最高):")
    print(best_preservation[['method', 'high_value_compression_ratio']])

    print("\n最佳压缩方法 (总体压缩率最低):")
    print(best_compression[['method', 'compression_ratio']])


def interactive_compression_visualization():
    """交互式压缩可视化"""
    results_dir = project_root / "results"

    # 获取可用的压缩文件
    compressed_files = list(results_dir.glob("*_voxel_0.1.pcd"))

    if not compressed_files:
        print("未找到压缩结果文件")
        return

    # 提取序列和文件信息
    file_info = []
    for f in compressed_files:
        name = f.stem
        parts = name.split('_')
        if len(parts) >= 2:
            seq_id = parts[0]  # 假设文件名格式为 000000_high_value_voxel_0.1.pcd
            file_info.append((seq_id, f))

    print("可用的压缩文件:")
    for i, (seq_id, f) in enumerate(file_info[:10]):  # 显示前10个
        print(f"{i}: {f.name}")

    if len(file_info) > 10:
        print(f"... 还有 {len(file_info) - 10} 个文件")

    try:
        choice = input("选择文件索引 (默认 0): ").strip()
        file_index = int(choice) if choice.isdigit() else 0

        if file_index < len(file_info):
            seq_id, selected_file = file_info[file_index]
            visualize_single_compression(selected_file)
        else:
            print("无效的选择")
    except Exception as e:
        print(f"输入错误: {e}")


def visualize_single_compression(compressed_file):
    """可视化单个压缩文件及其原始对应文件"""
    compressed_pcd = o3d.io.read_point_cloud(str(compressed_file))

    # 查找原始高价值文件
    file_name = compressed_file.stem
    base_parts = file_name.split('_high_value_')[0]
    original_name = f"{base_parts}_high_value"

    # 在可能的数据目录中查找
    possible_dirs = [
        project_root / "data" / "high_value_dataset_fast",
        project_root / "data" / "high_value_dataset"
    ]

    original_pcd = None
    for data_dir in possible_dirs:
        for seq_dir in data_dir.iterdir():
            if seq_dir.is_dir():
                original_file = seq_dir / f"{original_name}.pcd"
                if original_file.exists():
                    original_pcd = o3d.io.read_point_cloud(str(original_file))
                    break
        if original_pcd is not None:
            break

    if original_pcd is None:
        print(f"未找到原始文件: {original_name}")
        # 只显示压缩结果
        o3d.visualization.draw_geometries([compressed_pcd],
                                          window_name=f"压缩结果 - {compressed_file.name}")
        return

    # 设置颜色以便区分
    original_pcd.paint_uniform_color([1, 1, 1])  # 白色 - 原始
    compressed_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 压缩

    print(f"原始点数: {len(original_pcd.points)}")
    print(f"压缩点数: {len(compressed_pcd.points)}")
    print(f"压缩率: {len(compressed_pcd.points) / len(original_pcd.points):.2%}")

    o3d.visualization.draw_geometries(
        [original_pcd, compressed_pcd],
        window_name=f"压缩对比 - {compressed_file.name}",
        width=1000,
        height=800
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='压缩结果可视化')
    parser.add_argument('--compare', action='store_true', help='对比所有压缩方法')
    parser.add_argument('--stats', action='store_true', help='显示统计信息')
    parser.add_argument('--interactive', action='store_true', help='交互式可视化')
    parser.add_argument('--sequence', default="00", help='序列ID')
    parser.add_argument('--file_index', type=int, default=0, help='文件索引')

    args = parser.parse_args()

    if args.stats:
        show_compression_statistics()
    elif args.interactive:
        interactive_compression_visualization()
    elif args.compare:
        visualize_compression_comparison(args.sequence, args.file_index)
    else:
        # 默认显示统计信息
        show_compression_statistics()
        print("\n使用 --compare 查看可视化对比")
        print("使用 --interactive 进行交互式查看")