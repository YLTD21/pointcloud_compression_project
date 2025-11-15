# scripts/high_value_video_player.py
import open3d as o3d
import numpy as np
from pathlib import Path
import time
import os
import sys

project_root = Path(__file__).parent
sys.path.append(str(project_root))


class HighValueVideoPlayer:
    def __init__(self):
        self.vis = None
        self.pcd = o3d.geometry.PointCloud()
        self.current_frame = 0
        self.total_frames = 0
        self.files = []

    def load_high_value_sequence(self, sequence_id="00"):
        """加载高价值点云序列"""
        high_value_dir = project_root / "data" / "high_value_dataset" / f"seq_{sequence_id}"

        if not high_value_dir.exists():
            print(f"高价值序列 {sequence_id} 不存在")
            return False

        # 获取所有高价值点云文件
        self.files = [f for f in high_value_dir.iterdir()
                      if f.is_file() and f.suffix == '.pcd' and '_high_value' in f.stem]
        self.files.sort(key=lambda x: x.name)

        if not self.files:
            print(f"序列 {sequence_id} 没有高价值点云文件")
            return False

        self.total_frames = len(self.files)
        self.current_frame = 0

        print(f"加载序列 {sequence_id}: {self.total_frames} 个高价值点云文件")
        return True

    def load_fast_high_value_sequence(self, sequence_id="00"):
        """加载快速高价值点云序列"""
        high_value_dir = project_root / "data" / "high_value_dataset_fast" / f"seq_{sequence_id}"

        if not high_value_dir.exists():
            print(f"快速高价值序列 {sequence_id} 不存在")
            return False

        # 获取所有高价值点云文件
        self.files = [f for f in high_value_dir.iterdir()
                      if f.is_file() and f.suffix == '.pcd' and '_high_value' in f.stem]
        self.files.sort(key=lambda x: x.name)

        if not self.files:
            print(f"序列 {sequence_id} 没有高价值点云文件")
            return False

        self.total_frames = len(self.files)
        self.current_frame = 0

        print(f"加载快速序列 {sequence_id}: {self.total_frames} 个高价值点云文件")
        return True

    def load_frame(self, frame_index):
        """加载指定帧"""
        if frame_index < 0 or frame_index >= self.total_frames:
            return False

        file_path = self.files[frame_index]
        try:
            frame_pcd = o3d.io.read_point_cloud(str(file_path))

            if len(frame_pcd.points) == 0:
                print(f"帧 {frame_index} 为空点云")
                return False

            self.pcd.points = frame_pcd.points
            self.pcd.colors = frame_pcd.colors
            self.current_frame = frame_index

            print(f"帧 {frame_index}/{self.total_frames - 1}: {len(frame_pcd.points)} 个点")
            return True

        except Exception as e:
            print(f"加载帧 {frame_index} 失败: {e}")
            return False

    def play_sequence(self, sequence_id="00", frame_range=None, fps=5, point_size=3.0):
        """播放高价值点云序列"""
        if not self.load_high_value_sequence(sequence_id):
            return False

        # 设置帧范围
        start_frame = 0
        end_frame = self.total_frames - 1
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(0, min(end_frame, self.total_frames - 1))

        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=f"高价值点云播放器 - 序列 {sequence_id}",
            width=1200,
            height=800
        )

        # 添加点云
        self.vis.add_geometry(self.pcd)

        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景

        # 播放循环
        frame_delay = 1.0 / fps

        for frame_idx in range(start_frame, end_frame + 1):
            if not self.load_frame(frame_idx):
                continue

            # 更新点云
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            # 设置视角（第一帧时重置）
            if frame_idx == start_frame:
                self.vis.reset_view_point(True)

            # 显示帧信息
            print(f"播放: 帧 {frame_idx}/{end_frame} - {len(self.pcd.points)} 点")

            time.sleep(frame_delay)

            # 检查窗口是否关闭
            if not self.vis.poll_events():
                break

        # 关闭窗口
        self.vis.destroy_window()
        return True

    def play_fast_sequence(self, sequence_id="00", frame_range=None, fps=5, point_size=3.0):
        """播放快速高价值点云序列"""
        if not self.load_fast_high_value_sequence(sequence_id):
            return False

        # 设置帧范围
        start_frame = 0
        end_frame = self.total_frames - 1
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(0, min(end_frame, self.total_frames - 1))

        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=f"快速高价值点云播放器 - 序列 {sequence_id}",
            width=1200,
            height=800
        )

        # 添加点云
        self.vis.add_geometry(self.pcd)

        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景

        # 播放循环
        frame_delay = 1.0 / fps

        for frame_idx in range(start_frame, end_frame + 1):
            if not self.load_frame(frame_idx):
                continue

            # 更新点云
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            # 设置视角（第一帧时重置）
            if frame_idx == start_frame:
                self.vis.reset_view_point(True)

            # 显示帧信息
            print(f"播放: 帧 {frame_idx}/{end_frame} - {len(self.pcd.points)} 点")

            time.sleep(frame_delay)

            # 检查窗口是否关闭
            if not self.vis.poll_events():
                break

        # 关闭窗口
        self.vis.destroy_window()
        return True


def main():
    """主函数 - 高价值点云视频播放器"""
    player = HighValueVideoPlayer()

    print("高价值点云视频播放器")
    print("=" * 40)

    while True:
        print("\n选择播放模式:")
        print("1. 标准高价值点云序列")
        print("2. 快速高价值点云序列")
        print("3. 退出")

        choice = input("请选择 (1-3): ").strip()

        if choice == '3':
            break
        elif choice in ['1', '2']:
            sequence_id = input("输入序列ID (例如: 00): ").strip()
            fps = input("输入帧率 (默认 5): ").strip()
            fps = int(fps) if fps.isdigit() else 5

            point_size = input("输入点大小 (默认 3.0): ").strip()
            point_size = float(point_size) if point_size.replace('.', '').isdigit() else 3.0

            start_frame = input("输入起始帧 (默认 0): ").strip()
            start_frame = int(start_frame) if start_frame.isdigit() else 0

            end_frame = input("输入结束帧 (默认 100): ").strip()
            end_frame = int(end_frame) if end_frame.isdigit() else 100

            if choice == '1':
                player.play_sequence(
                    sequence_id=sequence_id,
                    frame_range=(start_frame, end_frame),
                    fps=fps,
                    point_size=point_size
                )
            else:
                player.play_fast_sequence(
                    sequence_id=sequence_id,
                    frame_range=(start_frame, end_frame),
                    fps=fps,
                    point_size=point_size
                )
        else:
            print("无效选择")


if __name__ == "__main__":
    main()