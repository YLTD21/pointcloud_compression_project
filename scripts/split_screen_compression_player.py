# scripts/split_screen_compression_player.py
import open3d as o3d
import numpy as np
from pathlib import Path
import time
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SplitScreenCompressionPlayer:
    def __init__(self):
        self.pcds = {}
        self.current_frame = 0
        self.total_frames = 0
        self.files_cache = {}
        self.methods = [
            'voxel_0.05', 'voxel_0.1', 'voxel_0.2',
            'statistical_10_1.5', 'statistical_20_2.0'
        ]
        self.method_names = {
            'voxel_0.05': 'Voxel 0.05',
            'voxel_0.1': 'Voxel 0.1',
            'voxel_0.2': 'Voxel 0.2',
            'statistical_10_1.5': 'Statistical 10/1.5',
            'statistical_20_2.0': 'Statistical 20/2.0'
        }
        # 为不同方法分配不同的颜色
        self.method_colors = {
            'voxel_0.05': [1, 0, 0],  # 红色
            'voxel_0.1': [0, 1, 0],  # 绿色
            'voxel_0.2': [0, 0, 1],  # 蓝色
            'statistical_10_1.5': [1, 1, 0],  # 黄色
            'statistical_20_2.0': [1, 0, 1]  # 紫色
        }

    def play_multiple_windows(self, sequence_id="00", frame_range=None, fps=2, point_size=2.0):
        """在5个独立窗口中同时播放视频"""
        if not self.load_compression_sequence(sequence_id):
            return False

        # 设置帧范围
        start_frame = 0
        end_frame = self.total_frames - 1
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(0, min(end_frame, self.total_frames - 1))

        # 创建5个窗口
        self.create_multiple_windows(point_size)

        print(f"\n开始在5个窗口中同时播放 {len(self.methods)} 种压缩方法...")
        print(f"帧率: {fps} FPS, 总帧数: {end_frame - start_frame + 1}")
        print("=" * 60)
        print("窗口布局和颜色说明:")
        print("┌─────────────┬─────────────┬─────────────┐")
        print("│ Voxel 0.05  │  Voxel 0.1  │  Voxel 0.2  │")
        print("│   红色      │    绿色     │    蓝色     │")
        print("├─────────────┼─────────────┼─────────────┤")
        print("│Statistical  │Statistical  │             │")
        print("│  10/1.5     │  20/2.0     │   空白区域   │")
        print("│   黄色      │    紫色     │             │")
        print("└─────────────┴─────────────┴─────────────┘")
        print("=" * 60)

        # 播放循环
        frame_delay = 1.0 / fps

        try:
            for frame_idx in range(start_frame, end_frame + 1):
                # 更新所有窗口
                success, point_counts = self.update_all_windows(frame_idx)
                if not success:
                    print(f"帧 {frame_idx} 加载失败，跳过")
                    continue

                # 显示帧信息
                print(f"\n帧 {frame_idx}/{end_frame} - 各方法点数:")
                for method in self.methods:
                    count = point_counts.get(method, 0)
                    color_name = self._get_color_name(method)
                    print(f"  {self.method_names[method]:<20} {color_name:<8} {count:>6} 点")

                # 检查是否有窗口关闭
                all_windows_active = True
                for method, window_info in self.windows.items():
                    if not window_info['vis'].poll_events():
                        all_windows_active = False
                        break

                if not all_windows_active:
                    print("检测到窗口关闭，停止播放")
                    break

                time.sleep(frame_delay)

        except KeyboardInterrupt:
            print("\n播放被用户中断")
        except Exception as e:
            print(f"\n播放过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 关闭所有窗口
            self.close_all_windows()
            print("\n所有窗口已关闭")

        return True
    def load_compression_sequence(self, sequence_id="00"):
        """加载所有压缩方法的序列文件"""
        results_dir = project_root / "results"

        if not results_dir.exists():
            print(f"结果目录不存在: {results_dir}")
            return False

        # 查找所有压缩文件
        all_files = {}
        for method in self.methods:
            # 确保使用正确的文件模式
            method_files = list(results_dir.glob(f"*{method}.pcd"))
            if not method_files:
                # 尝试其他文件模式
                method_files = list(results_dir.glob(f"*{method}*.pcd"))

            method_files.sort(key=lambda x: x.name)
            all_files[method] = method_files

            print(f"方法 {method}: 找到 {len(method_files)} 个文件")

        # 检查文件数量一致性
        file_counts = [len(files) for files in all_files.values()]
        if len(set(file_counts)) != 1:
            print("警告: 不同方法的文件数量不一致")
            for method, files in all_files.items():
                print(f"  {method}: {len(files)} 个文件")

        # 使用最小文件数量
        self.total_frames = min(file_counts) if file_counts else 0

        if self.total_frames == 0:
            print("未找到压缩文件")
            print("请确保已运行第三步压缩流程")
            return False

        self.files_cache = all_files
        print(f"加载序列 {sequence_id}: {self.total_frames} 帧")

        # 显示找到的文件示例
        for method in self.methods:
            if self.files_cache[method]:
                print(f"  {method}: 示例文件 {self.files_cache[method][0].name}")

        return True

    def load_frame_for_method(self, method, frame_index):
        """为指定方法加载指定帧的点云并设置颜色"""
        if method not in self.files_cache or frame_index >= len(self.files_cache[method]):
            return None

        file_path = self.files_cache[method][frame_index]
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            if len(pcd.points) > 0:
                # 为点云设置对应方法的颜色
                colors = np.zeros((len(pcd.points), 3))
                colors[:] = self.method_colors[method]
                pcd.colors = o3d.utility.Vector3dVector(colors)
                return pcd
            else:
                print(f"警告: {method} 帧 {frame_index} 点云为空")
        except Exception as e:
            print(f"加载 {method} 帧 {frame_index} 失败: {e}")
        return None

    def create_multiple_windows(self, point_size=2.0):
        """创建5个独立窗口"""
        self.windows = {}

        # 窗口位置计算 - 适应不同屏幕
        screen_width, screen_height = 1920, 1080
        window_width, window_height = 500, 400

        # 5个窗口的位置 (2行3列布局)
        positions = [
            (50, screen_height - window_height - 50),  # 左上: voxel_0.05
            (window_width + 70, screen_height - window_height - 50),  # 中上: voxel_0.1
            (window_width * 2 + 90, screen_height - window_height - 50),  # 右上: voxel_0.2
            (50, screen_height - window_height * 2 - 70),  # 左下: statistical_10_1.5
            (window_width + 70, screen_height - window_height * 2 - 70)  # 中下: statistical_20_2.0
        ]

        # 定义不同的相机视角参数
        camera_parameters = [
            # front: 前视角, lookat: 注视点, up: 上方向, zoom: 缩放
            {'front': [0, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.1},  # voxel_0.05
            {'front': [0, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.1},  # voxel_0.1
            {'front': [0, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.1},  # voxel_0.2
            {'front': [0, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.1},  # statistical_10_1.5
            {'front': [0, -1, -0.5], 'lookat': [0, 0, 0], 'up': [0, -1, 0], 'zoom': 0.1},  # statistical_20_2.0
        ]

        for i, method in enumerate(self.methods):
            # 创建可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{self.method_names[method]}",
                width=window_width,
                height=window_height,
                left=positions[i][0],
                top=positions[i][1],
                visible=True
            )

            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.array([0.05, 0.05, 0.05])

            self.windows[method] = {
                'vis': vis,
                'position': positions[i],
                'current_pcd': None,
                'window_id': i + 1,
                'camera_params': camera_parameters[i]  # 保存相机参数
            }

            print(f"创建窗口 {i + 1}: {self.method_names[method]} - 颜色 {self.method_colors[method]}")
    def close_all_windows(self):
        """关闭所有窗口"""
        for method, window_info in self.windows.items():
            try:
                window_info['vis'].destroy_window()
            except:
                pass
        self.windows.clear()

    def update_all_windows(self, frame_index):
        """更新所有窗口的点云"""
        all_success = True
        point_counts = {}

        for method, window_info in self.windows.items():
            vis = window_info['vis']
            view_control = vis.get_view_control()

            # 加载当前帧的点云（带颜色）
            new_pcd = self.load_frame_for_method(method, frame_index)
            if new_pcd is None:
                all_success = False
                continue

            point_counts[method] = len(new_pcd.points)

            # 移除旧的点云（如果存在）
            if window_info['current_pcd'] is not None:
                vis.remove_geometry(window_info['current_pcd'], reset_bounding_box=False)

            # 添加新的点云
            vis.add_geometry(new_pcd, reset_bounding_box=(frame_index == 0))
            window_info['current_pcd'] = new_pcd

            # 设置相机视角（只在第一帧或需要时设置）
            if frame_index == 0:
                self.set_low_angle_camera(view_control, window_info['camera_params'])

            # 更新渲染
            vis.poll_events()
            vis.update_renderer()

        return all_success, point_counts

    def set_low_angle_camera(self, view_control, camera_params):
        """设置低角度相机视角"""
        try:
            # 方法1: 使用set_lookat设置注视点
            view_control.set_lookat(camera_params['lookat'])

            # 方法2: 使用set_front设置前向向量（从低角度观察）
            view_control.set_front(camera_params['front'])

            # 方法3: 设置上方向
            view_control.set_up(camera_params['up'])

            # 方法4: 设置缩放
            view_control.set_zoom(camera_params['zoom'])

        except Exception as e:
            print(f"设置相机视角失败: {e}")
            # 备用方案：使用reset_view_point
            view_control.reset_view_point(True)

    def play_individual_methods(self, sequence_id="00", frame_range=None, fps=5, point_size=3.0):
        """分别播放每个压缩方法（顺序播放）"""
        if not self.load_compression_sequence(sequence_id):
            return False

        # 设置帧范围
        start_frame = 0
        end_frame = self.total_frames - 1
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(0, min(end_frame, self.total_frames - 1))

        for method in self.methods:
            print(f"\n开始播放: {self.method_names[method]}")
            print(f"颜色: {self._get_color_name(method)}")

            # 创建单独的可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{self.method_names[method]} - {self._get_color_name(method)} - 序列 {sequence_id}",
                width=1000,
                height=800
            )

            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.array([0.05, 0.05, 0.05])

            current_pcd = None
            view_control = vis.get_view_control()

            # 播放循环
            frame_delay = 1.0 / fps

            try:
                for frame_idx in range(start_frame, end_frame + 1):
                    # 加载当前帧（带颜色）
                    new_pcd = self.load_frame_for_method(method, frame_idx)
                    if new_pcd is None:
                        continue

                    # 移除旧的点云
                    if current_pcd is not None:
                        vis.remove_geometry(current_pcd, reset_bounding_box=False)

                    # 添加新的点云
                    vis.add_geometry(new_pcd, reset_bounding_box=(frame_idx == start_frame))
                    current_pcd = new_pcd

                    # 设置低角度相机视角（只在第一帧设置）
                    if frame_idx == start_frame:
                        low_angle_params = {
                            'front': [0, -1, -0.3],  # 更低的角度
                            'lookat': [0, 0, 0],
                            'up': [0, -1, 0],
                            'zoom': 0.6  # 更近的视角
                        }
                        self.set_low_angle_camera(view_control, low_angle_params)

                    # 更新渲染
                    vis.poll_events()
                    vis.update_renderer()

                    # 显示帧信息
                    point_count = len(new_pcd.points)
                    print(f"帧 {frame_idx}/{end_frame} - {point_count} 点")

                    time.sleep(frame_delay)

                    # 检查窗口是否关闭
                    if not vis.poll_events():
                        break

            except KeyboardInterrupt:
                print(f"{method} 播放被用户中断")
            except Exception as e:
                print(f"{method} 播放过程中出错: {e}")
            finally:
                # 关闭当前窗口
                vis.destroy_window()

            # 询问是否继续播放下一个方法
            if method != self.methods[-1]:
                next_method = self.methods[self.methods.index(method) + 1]
                next_color = self._get_color_name(next_method)
                response = input(f"\n继续播放下一个方法 {next_method} ({next_color})? (y/n): ")
                if response.lower() != 'y':
                    break

        return True

    def _get_color_name(self, method):
        """获取颜色名称"""
        color_map = {
            'voxel_0.05': '红色',
            'voxel_0.1': '绿色',
            'voxel_0.2': '蓝色',
            'statistical_10_1.5': '黄色',
            'statistical_20_2.0': '紫色'
        }
        return color_map.get(method, '未知')

    def play_individual_methods(self, sequence_id="00", frame_range=None, fps=5, point_size=3.0):
        """分别播放每个压缩方法（顺序播放）"""
        if not self.load_compression_sequence(sequence_id):
            return False

        # 设置帧范围
        start_frame = 0
        end_frame = self.total_frames - 1
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(0, min(end_frame, self.total_frames - 1))

        for method in self.methods:
            print(f"\n开始播放: {self.method_names[method]}")
            print(f"颜色: {self._get_color_name(method)}")

            # 创建单独的可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{self.method_names[method]} - {self._get_color_name(method)} - 序列 {sequence_id}",
                width=1000,
                height=800
            )

            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            render_option.background_color = np.array([0.05, 0.05, 0.05])

            current_pcd = None

            # 播放循环
            frame_delay = 1.0 / fps

            try:
                for frame_idx in range(start_frame, end_frame + 1):
                    # 加载当前帧（带颜色）
                    new_pcd = self.load_frame_for_method(method, frame_idx)
                    if new_pcd is None:
                        continue

                    # 移除旧的点云
                    if current_pcd is not None:
                        vis.remove_geometry(current_pcd, reset_bounding_box=False)

                    # 添加新的点云
                    vis.add_geometry(new_pcd, reset_bounding_box=(frame_idx == start_frame))
                    current_pcd = new_pcd

                    # 更新渲染
                    vis.poll_events()
                    vis.update_renderer()

                    # 显示帧信息
                    point_count = len(new_pcd.points)
                    print(f"帧 {frame_idx}/{end_frame} - {point_count} 点")

                    time.sleep(frame_delay)

                    # 检查窗口是否关闭
                    if not vis.poll_events():
                        break

            except KeyboardInterrupt:
                print(f"{method} 播放被用户中断")
            except Exception as e:
                print(f"{method} 播放过程中出错: {e}")
            finally:
                # 关闭当前窗口
                vis.destroy_window()

            # 询问是否继续播放下一个方法
            if method != self.methods[-1]:
                next_method = self.methods[self.methods.index(method) + 1]
                next_color = self._get_color_name(next_method)
                response = input(f"\n继续播放下一个方法 {next_method} ({next_color})? (y/n): ")
                if response.lower() != 'y':
                    break

        return True

    def verify_compression_files(self):
        """验证压缩文件是否正确"""
        print("验证压缩文件...")
        results_dir = project_root / "results"

        if not results_dir.exists():
            print("结果目录不存在")
            return False

        for method in self.methods:
            files = list(results_dir.glob(f"*{method}.pcd"))
            print(f"{method}: {len(files)} 个文件")
            if files:
                # 检查第一个文件的内容
                try:
                    pcd = o3d.io.read_point_cloud(str(files[0]))
                    print(f"  - 示例文件: {files[0].name}")
                    print(f"  - 点数: {len(pcd.points)}")
                    print(f"  - 是否有颜色: {pcd.has_colors()}")
                except Exception as e:
                    print(f"  - 读取失败: {e}")

        return True


def main():
    """主函数 - 多窗口压缩视频播放器"""
    player = SplitScreenCompressionPlayer()

    print("多窗口压缩对比播放器")
    print("=" * 60)

    # 验证文件
    player.verify_compression_files()

    while True:
        print("\n选择播放模式:")
        print("1. 多窗口同时播放 (5个独立窗口同步视频，不同颜色)")
        print("2. 分别顺序播放 (单个窗口顺序播放，带颜色)")
        print("3. 验证压缩文件")
        print("4. 退出")

        choice = input("请选择 (1-4): ").strip()

        if choice == '4':
            break
        elif choice in ['1', '2', '3']:
            sequence_id = input("输入序列ID (默认 00): ").strip() or "00"

            if choice == '1':
                fps = input("输入帧率 (默认 2): ").strip()
                fps = int(fps) if fps.isdigit() else 2

                point_size = input("输入点大小 (默认 2.0): ").strip()
                point_size = float(point_size) if point_size.replace('.', '').isdigit() else 2.0

                start_frame = input("输入起始帧 (默认 0): ").strip()
                start_frame = int(start_frame) if start_frame.isdigit() else 0

                end_frame = input("输入结束帧 (默认 50): ").strip()
                end_frame = int(end_frame) if end_frame.isdigit() else 50

                player.play_multiple_windows(
                    sequence_id=sequence_id,
                    frame_range=(start_frame, end_frame),
                    fps=fps,
                    point_size=point_size
                )

            elif choice == '2':
                fps = input("输入帧率 (默认 5): ").strip()
                fps = int(fps) if fps.isdigit() else 5

                point_size = input("输入点大小 (默认 3.0): ").strip()
                point_size = float(point_size) if point_size.replace('.', '').isdigit() else 3.0

                start_frame = input("输入起始帧 (默认 0): ").strip()
                start_frame = int(start_frame) if start_frame.isdigit() else 0

                end_frame = input("输入结束帧 (默认 50): ").strip()
                end_frame = int(end_frame) if end_frame.isdigit() else 50

                player.play_individual_methods(
                    sequence_id=sequence_id,
                    frame_range=(start_frame, end_frame),
                    fps=fps,
                    point_size=point_size
                )

            elif choice == '3':
                player.verify_compression_files()

        else:
            print("无效选择")


if __name__ == "__main__":
    main()