# scripts/main.py
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def print_menu():
    """打印主菜单"""
    print("\n" + "=" * 50)
    print("       点云处理流水线 - 主菜单")
    print("=" * 50)
    print("1. 提取行人和车辆数据（增强版）")
    print("2. 提取高价值特征点")
    print("3. 点云压缩")
    print("4. 执行完整流程")
    print("5. 测试单个文件")
    print("6. 查看处理结果统计")
    print("7. 检查数据目录结构")
    print("8. 可视化点云数据")
    print("9. 可视化高价值点云（视频）")  # 新增选项
    print("0. 退出")
    print("-" * 50)

# 添加对应的处理函数
def run_high_value_visualization():
    """运行高价值点云视频可视化"""
    print("\n>>> 启动高价值点云视频可视化")
    try:
        from high_value_video_player import main as high_value_main
        high_value_main()
    except Exception as e:
        print(f"高价值点云可视化启动失败: {e}")
        print("请确保已安装所有依赖")

def get_user_choice():
    """获取用户选择"""
    while True:
        try:
            choice = input("请选择要执行的操作 (0-8): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            else:
                print("无效选择，请输入 0-8 之间的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            sys.exit(0)
        except:
            print("无效输入，请重新选择")


def run_step1():
    """执行步骤1：提取行人和车辆数据（增强版）"""
    print("\n>>> 开始执行步骤1：提取行人和车辆数据（增强版）")

    try:
        from step1_extract_objects_final import process_semantic_kitti_data_final
        process_semantic_kitti_data_final()
        print("✓ 步骤1完成：行人和车辆数据提取成功")
        return True
    except Exception as e:
        print(f"✗ 步骤1执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_step2():
    """执行步骤2：提取高价值特征点"""
    print("\n>>> 开始执行步骤2：提取高价值特征点")

    # 检查步骤1的输出是否存在
    processed_dir = project_root / "data" / "processed_dataset_final"  # 更新路径
    pcd_files = list(processed_dir.rglob("*.pcd"))

    if not processed_dir.exists() or len(pcd_files) == 0:
        print("错误：未找到处理后的数据，请先执行步骤1")
        print(f"请检查目录: {processed_dir}")
        return False

    print(f"找到 {len(pcd_files)} 个处理后的文件")

    try:
        from step2_extract_high_value_features import process_high_value_extraction
        process_high_value_extraction()
        print("✓ 步骤2完成：高价值特征点提取成功")
        return True
    except Exception as e:
        print(f"✗ 步骤2执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_step3():
    """执行步骤3：点云压缩"""
    print("\n>>> 开始执行步骤3：点云压缩")

    # 检查步骤2的输出是否存在
    high_value_dir = project_root / "data" / "high_value_dataset"
    npy_files = list(high_value_dir.rglob("*_high_value.npy"))

    if not high_value_dir.exists() or len(npy_files) == 0:
        print("错误：未找到高价值特征数据，请先执行步骤2")
        print(f"请检查目录: {high_value_dir}")
        return False

    print(f"找到 {len(npy_files)} 个高价值特征文件")

    try:
        from step3_pointcloud_compression import process_compression
        process_compression()
        print("✓ 步骤3完成：点云压缩成功")
        return True
    except Exception as e:
        print(f"✗ 步骤3执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline():
    """执行完整流程"""
    print("\n>>> 开始执行完整处理流程")

    steps = [
        ("步骤1: 提取行人和车辆（增强版）", run_step1),
        ("步骤2: 提取高价值特征", run_step2),
        ("步骤3: 点云压缩", run_step3)
    ]

    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        start_time = time.time()
        success = step_func()
        end_time = time.time()

        if success:
            print(f"✓ {step_name} 完成 (耗时: {end_time - start_time:.2f}秒)")
        else:
            print(f"✗ {step_name} 失败，流程终止")
            return False

        # 步骤间暂停，让用户观察结果
        if step_name != "步骤3: 点云压缩":
            input("按Enter键继续下一步骤...")

    print("\n🎉 完整处理流程执行完成！")
    return True


def run_test():
    """测试单个文件"""
    print("\n>>> 开始测试单个文件")
    try:
        from test_pipeline import test_single_file
        test_single_file()
        print("✓ 测试完成")
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def show_statistics():
    """显示处理结果统计"""
    print("\n>>> 处理结果统计")

    # 检查各目录的文件数量
    directories = {
        "原始数据": project_root / "data" / "raw_dataset",
        "处理后数据（增强版）": project_root / "data" / "processed_dataset_final",  # 更新
        "高价值数据": project_root / "data" / "high_value_dataset",
        "压缩结果": project_root / "results"
    }

    for name, path in directories.items():
        if path.exists():
            if name == "原始数据":
                files = list(path.rglob("*.bin"))
            elif name == "处理后数据（增强版）":
                files = list(path.rglob("*.pcd"))
            elif name == "高价值数据":
                files = list(path.rglob("*_high_value.npy"))
            else:  # 压缩结果
                files = list(path.glob("*.pcd")) + list(path.glob("*.csv"))

            print(f"{name}: {len(files)} 个文件")
        else:
            print(f"{name}: 目录不存在")

    # 如果有压缩统计文件，显示压缩率
    stats_file = project_root / "results" / "compression_statistics.csv"
    if stats_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(stats_file)
            print("\n压缩统计:")
            print(df.groupby('method')[['compression_ratio', 'high_value_compression_ratio']].mean())
        except Exception as e:
            print(f"读取压缩统计文件失败: {e}")


def check_data_structure():
    """检查数据目录结构"""
    print("\n>>> 检查数据目录结构")

    from utils import find_semantic_kitti_sequences, get_project_root

    project_root = get_project_root()

    print(f"项目根目录: {project_root}")

    # 检查关键目录
    key_dirs = [
        ("原始数据目录", project_root / "data" / "raw_dataset"),
        ("处理后数据目录（增强版）", project_root / "data" / "processed_dataset_final"),  # 更新
        ("高价值数据目录", project_root / "data" / "high_value_dataset"),
        ("结果目录", project_root / "results")
    ]

    for name, path in key_dirs:
        if path.exists():
            print(f"✓ {name}: {path} (存在)")
            # 如果是原始数据目录，进一步检查内容
            if name == "原始数据目录":
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                print(f"  包含子目录: {[d.name for d in subdirs]}")
        else:
            print(f"✗ {name}: {path} (不存在)")

    # 查找SemanticKITTI序列
    print("\n查找SemanticKITTI数据序列...")
    sequences = find_semantic_kitti_sequences()

    if sequences:
        print(f"找到 {len(sequences)} 个数据序列:")
        for seq in sequences:
            print(f"  序列 {seq['seq_id']}:")
            print(f"    点云文件: {len(list(seq['velodyne_path'].glob('*.bin')))} 个")
            if seq['labels_path']:
                print(f"    标签文件: {len(list(seq['labels_path'].glob('*.label')))} 个")
            else:
                print(f"    标签文件: 未找到")
    else:
        print("未找到任何SemanticKITTI数据序列！")
        print("\n请确保数据目录结构为:")
        print("data/raw_dataset/dataset/sequences/00/velodyne/")
        print("data/raw_dataset/dataset/sequences/00/labels/")


def run_visualization():
    """运行可视化工具"""
    print("\n>>> 启动点云可视化工具")
    try:
        from enhanced_objects_player import main as visualize_main
        visualize_main()
    except Exception as e:
        print(f"可视化工具启动失败: {e}")
        print("请确保已安装所有依赖")


def main():
    """主函数"""
    print("点云处理系统初始化...")

    # 检查必要的目录
    required_dirs = [
        project_root / "data" / "processed_dataset_final",  # 更新
        project_root / "data" / "high_value_dataset",
        project_root / "results"
    ]

    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 主循环
    while True:
        print_menu()
        choice = get_user_choice()

        if choice == 0:
            print("感谢使用点云处理系统，再见！")
            break
        elif choice == 1:
            run_step1()
        elif choice == 2:
            run_step2()
        elif choice == 3:
            run_step3()
        elif choice == 4:
            run_full_pipeline()
        elif choice == 5:
            run_test()
        elif choice == 6:
            show_statistics()
        elif choice == 7:
            check_data_structure()
        elif choice == 8:
            run_visualization()
        elif choice == 9:  # 新增选项
            run_high_value_visualization()
        # 操作完成后暂停
        if choice != 0:
            input("\n按Enter键返回主菜单...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback

        traceback.print_exc()