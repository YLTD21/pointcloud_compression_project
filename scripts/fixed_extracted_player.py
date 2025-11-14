# scripts/fixed_extracted_player.py
import open3d as o3d
import numpy as np
import time
from pathlib import Path
import sys


class FixedExtractedPlayer:
    def __init__(self):
        self.vis = None
        self.pcd = None
        self.is_playing = False

    def play_sequence(self, sequence_id, frame_range=None, fps=3):
        """修复的提取对象播放器"""
        source_dir = Path(f"../data/processed_dataset/seq_{sequence_id}")

        if not source_dir.exists():
            print(f"错误：序列目录不存在: {source_dir}")
            return False

        pcd_files = sorted(source_dir.glob("*.pcd"))

        if frame_range:
            pcd_files = pcd_files[frame_range[0]:frame_range[1]]

        total_frames = len(pcd_files)

        if total_frames == 0:
            print(f"序列 {sequence_id} 中没有点云文件")
            return False

        print(f"播放序列 {sequence_id}: {total_frames} 帧")

        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=f"提取对象 - 序列 {sequence_id}",
            width=1200,
            height=800,
            left=50,
            top=50,
            visible=True  # 确保窗口可见
        )

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # 强制设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.point_size = 15.0  # 很大的点
        render_option.background_color = np.array([0, 0, 0])  # 纯黑背景
        render_option.light_on = True

        # 初始渲染
        self.vis.poll_events()
        self.vis.update_renderer()

        self.is_playing = True
        frame_delay = 1.0 / fps

        for frame_idx, file_path in enumerate(pcd_files):
            if not self.is_playing:
                break

            try:
                # 加载点云
                pcd_data = o3d.io.read_point_cloud(str(file_path))
                points = np.asarray(pcd_data.points)

                if len(points) == 0:
                    print(f"帧 {frame_idx}: 空点云，跳过")
                    continue

                # 设置颜色
                if pcd_data.has_colors():
                    colors = np.asarray(pcd_data.colors)
                else:
                    colors = self.create_bright_colors(points)

                # 更新点云
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

                # 动态调整点大小
                point_count = len(points)
                if point_count < 10:
                    render_option.point_size = 30.0
                elif point_count < 50:
                    render_option.point_size = 20.0
                else:
                    render_option.point_size = 10.0

                # 更新视角
                self.adjust_camera_view(points)

                # 更新可视化 - 多次更新确保渲染
                for _ in range(3):
                    self.vis.update_geometry(self.pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()

                print(f"播放帧 {frame_idx + 1}/{total_frames} - 点数: {point_count}")

                time.sleep(frame_delay)

            except Exception as e:
                print(f"播放帧 {frame_idx} 失败: {e}")
                continue

        self.vis.destroy_window()
        return True

    def adjust_camera_view(self, points):
        """调整相机视角"""
        if len(points) == 0:
            return

        view_control = self.vis.get_view_control()

        # 计算点云中心
        center = np.mean(points, axis=0)

        # 设置视角参数
        view_control.set_lookat(center)
        view_control.set_front([0, 0, -1])  # 从正前方看
        view_control.set_up([0, -1, 0])  # Y轴向上

        # 计算合适的缩放级别
        max_range = np.max(np.ptp(points, axis=0))
        if max_range > 0:
            zoom_level = 0.5 / max_range
        else:
            zoom_level = 0.1

        view_control.set_zoom(min(zoom_level, 5.0))

    def create_bright_colors(self, points):
        """创建明亮的颜色"""
        colors = np.ones((len(points), 3))

        if len(points) == 1:
            colors[0] = [1, 1, 0]  # 黄色
        elif len(points) <= 10:
            # 少量点用亮绿色
            colors[:, 0] = 0
            colors[:, 1] = 1
            colors[:, 2] = 0
        else:
            # 多个点用彩虹色
            for i in range(len(points)):
                # 简单的彩虹色分布
                hue = i / len(points)
                if hue < 0.33:
                    colors[i] = [1, hue * 3, 0]  # 红到黄
                elif hue < 0.66:
                    colors[i] = [1 - (hue - 0.33) * 3, 1, 0]  # 黄到绿
                else:
                    colors[i] = [0, 1, (hue - 0.66) * 3]  # 绿到青

        return colors


def main():
    print("=== 修复版提取对象播放器 ===")

    # 先运行调试
    from debug_extracted_objects import debug_extracted_objects
    debug_extracted_objects()

    print("\n开始播放...")
    player = FixedExtractedPlayer()

    # 播放序列00的前50帧
    success = player.play_sequence("00", frame_range=(0, 50), fps=3)

    if success:
        print("播放完成!")
    else:
        print("播放失败!")


if __name__ == "__main__":
    main()