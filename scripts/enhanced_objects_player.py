# scripts/enhanced_objects_player.py
import open3d as o3d
import numpy as np
import time
from pathlib import Path
import sys


class EnhancedObjectsPlayer:
    def __init__(self):
        self.vis = None
        self.pcd = None
        self.is_playing = False

    def play_enhanced_sequence(self, sequence_id, frame_range=None, fps=5,
                               point_size=3.0, background_color=[0.05, 0.05, 0.1]):
        """
        播放增强版点云序列（包含背景的行人和车辆）
        """
        source_dir = Path(f"../data/processed_dataset_final/seq_{sequence_id}")

        if not source_dir.exists():
            print(f"增强版序列不存在: {source_dir}")
            return False

        # 获取所有点云文件
        files = sorted(source_dir.glob("*.pcd"))

        if frame_range:
            files = files[frame_range[0]:frame_range[1]]

        total_frames = len(files)

        if total_frames == 0:
            print(f"序列 {sequence_id} 中没有点云文件")
            return False

        print(f"准备播放增强版序列 {sequence_id}: {total_frames} 帧, {fps} FPS")

        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=f"增强版点云 - 序列 {sequence_id}",
            width=1400,
            height=900,
            left=50,
            top=50
        )

        self.pcd = o3d.geometry.PointCloud()

        # 加载第一帧
        if files:
            first_pcd = o3d.io.read_point_cloud(str(files[0]))
            self.pcd.points = first_pcd.points
            self.pcd.colors = first_pcd.colors

        self.vis.add_geometry(self.pcd)

        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(background_color)
        render_option.point_size = point_size
        render_option.light_on = True

        # 设置视角
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.3)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])

        # 初始渲染
        self.vis.poll_events()
        self.vis.update_renderer()

        print("开始播放增强版点云序列...")

        # 开始播放循环
        self.play_enhanced_loop(files, fps)

        self.vis.destroy_window()
        return True

    def play_enhanced_loop(self, files, fps):
        """增强版播放循环"""
        frame_delay = 1.0 / fps
        self.is_playing = True

        for frame_idx, file_path in enumerate(files):
            if not self.is_playing:
                break

            start_time = time.time()

            try:
                # 加载点云
                pcd_data = o3d.io.read_point_cloud(str(file_path))
                points = np.asarray(pcd_data.points)
                colors = np.asarray(pcd_data.colors)

                if len(points) == 0:
                    print(f"帧 {frame_idx}: 空点云，跳过")
                    continue

                # 更新点云
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

                # 统计信息
                red_points = np.sum(np.all(np.isclose(colors, [1, 0, 0], atol=0.1), axis=1))  # 车辆
                green_points = np.sum(np.all(np.isclose(colors, [0, 1, 0], atol=0.1), axis=1))  # 行人

                # 更新可视化
                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

                # 显示进度和统计
                progress = (frame_idx + 1) / len(files) * 100
                print(f"\r进度: {frame_idx + 1}/{len(files)} ({progress:.1f}%) - "
                      f"总点数: {len(points)} - 车辆: {red_points} - 行人: {green_points}",
                      end="", flush=True)

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                print(f"播放帧 {frame_idx} 时出错: {e}")
                continue

        self.is_playing = False
        print("\n播放结束")

    def stop(self):
        """停止播放"""
        self.is_playing = False


def play_all_enhanced_sequences(start_seq=0, end_seq=10, frames_per_sequence=100, fps=5):
    """播放所有增强版序列"""
    print(f"=== 播放所有增强版序列 (序列{start_seq:02d}到{end_seq:02d}) ===")

    player = EnhancedObjectsPlayer()

    for seq_id in range(start_seq, end_seq + 1):
        seq_str = f"{seq_id:02d}"

        # 检查序列是否存在
        source_dir = Path(f"../data/processed_dataset_enhanced/seq_{seq_str}")
        if not source_dir.exists():
            print(f"增强版序列 {seq_str} 不存在，跳过")
            continue

        # 检查是否有PCD文件
        pcd_files = list(source_dir.glob("*.pcd"))
        if not pcd_files:
            print(f"增强版序列 {seq_str} 没有点云文件，跳过")
            continue

        total_frames = len(pcd_files)
        frames_to_play = min(frames_per_sequence, total_frames)

        print(f"\n正在播放增强版序列 {seq_str} ({frames_to_play}/{total_frames} 帧)...")

        # 播放这个序列
        success = player.play_enhanced_sequence(
            sequence_id=seq_str,
            frame_range=(0, frames_to_play),
            fps=fps,
            point_size=3.0
        )

        if not success:
            print(f"增强版序列 {seq_str} 播放失败")

        # 如果不是最后一个序列，询问是否继续
        if seq_id < end_seq:
            response = input(f"\n序列 {seq_str} 播放完成。继续播放下一个序列? (y/n): ")
            if response.lower() != 'y':
                print("停止播放")
                break


def compare_original_vs_enhanced(sequence_id="00", frame_range=(0, 50)):
    """比较原始点云和增强版点云"""
    from pointcloud_video_player import PointCloudVideoPlayer  # 导入原始播放器

    print(f"=== 比较序列 {sequence_id} 的原始点云 vs 增强版点云 ===")

    # 先播放原始点云
    print("\n1. 播放原始点云...")
    original_player = PointCloudVideoPlayer()
    original_player.create_smooth_video(
        sequence_id=sequence_id,
        frame_range=frame_range,
        fps=10,
        point_size=1.0
    )

    # 再播放增强版点云
    print("\n2. 播放增强版点云（带背景的行人和车辆）...")
    enhanced_player = EnhancedObjectsPlayer()
    enhanced_player.play_enhanced_sequence(
        sequence_id=sequence_id,
        frame_range=frame_range,
        fps=5,
        point_size=3.0
    )


if __name__ == "__main__":
    print("=== 增强版点云视频播放器 ===")
    print("1. 播放单个增强版序列")
    print("2. 播放所有增强版序列 (00-10)")
    print("3. 比较原始点云 vs 增强版点云")
    print("4. 退出")

    choice = input("请选择: ").strip()

    if choice == "1":
        seq_id = input("输入序列ID (00-10): ").strip()
        frames = int(input("播放多少帧? (默认100): ") or "100")
        fps = int(input("帧率? (默认5): ") or "5")

        player = EnhancedObjectsPlayer()
        player.play_enhanced_sequence(
            sequence_id=seq_id,
            frame_range=(0, frames),
            fps=fps
        )

    elif choice == "2":
        start_seq = int(input("起始序列? (0-10, 默认0): ") or "0")
        end_seq = int(input("结束序列? (0-10, 默认10): ") or "10")
        frames_per_seq = int(input("每序列播放多少帧? (默认100): ") or "100")
        fps = int(input("帧率? (默认5): ") or "5")

        play_all_enhanced_sequences(start_seq, end_seq, frames_per_seq, fps)

    elif choice == "3":
        seq_id = input("输入序列ID (默认00): ").strip() or "00"
        frames = int(input("播放多少帧? (默认50): ") or "50")

        compare_original_vs_enhanced(seq_id, (0, frames))

    elif choice == "4":
        print("退出")

    else:
        print("无效选择")
#该代码用来查看提取好的，只有车辆和行人的点云代码，会生成视频而不是单帧