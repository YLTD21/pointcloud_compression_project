import open3d as o3d
import numpy as np
import time
from pathlib import Path
import threading

class PointCloudVideoPlayer:
    def __init__(self):
        self.vis = None
        self.pcd = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.playback_speed = 1.0
        self.pause_flag = False

    def create_smooth_video(self, sequence_id, frame_range=None, fps=10,
                          point_size=1.0, background_color=[0, 0, 0]):
        """
        创建流畅的点云视频播放器
        """
        source_dir = Path(f"../data/raw_dataset/dataset/sequences/{sequence_id}/velodyne")

        if not source_dir.exists():
            print(f"序列不存在: {source_dir}")
            return

        # 获取所有点云文件
        files = sorted(source_dir.glob("*.bin"))

        if frame_range:
            files = files[frame_range[0]:frame_range[1]]

        self.total_frames = len(files)

        if self.total_frames == 0:
            print("没有找到点云文件")
            return

        print(f"准备播放序列 {sequence_id}: {self.total_frames} 帧, {fps} FPS")

        # 预加载几帧数据到内存（为了流畅播放）
        print("预加载点云数据...")
        preloaded_frames = self.preload_frames(files, max_frames=min(100, self.total_frames))

        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=f"点云视频 - 序列 {sequence_id}",
            width=1400,
            height=900,
            left=50,
            top=50
        )

        self.pcd = o3d.geometry.PointCloud()

        # 设置初始点云（第一帧）
        if preloaded_frames:
            points, colors = preloaded_frames[0]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(self.pcd)

        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(background_color)
        render_option.point_size = point_size
        render_option.light_on = True

        # 设置视角
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.4)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])

        # 初始渲染
        self.vis.poll_events()
        self.vis.update_renderer()

        print("开始播放... (按ESC退出, 空格暂停/继续)")

        # 开始播放循环
        self.play_video_loop(files, fps, preloaded_frames)

        self.vis.destroy_window()

    def preload_frames(self, files, max_frames=100):
        """预加载帧数据到内存以提高播放性能"""
        preloaded = []

        for i, file_path in enumerate(files[:max_frames]):
            try:
                points_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                points = points_data[:, :3]
                colors = self.color_by_elevation(points)
                preloaded.append((points, colors))
            except Exception as e:
                print(f"预加载帧 {file_path} 失败: {e}")

        print(f"预加载了 {len(preloaded)} 帧")
        return preloaded

    def play_video_loop(self, files, fps, preloaded_frames):
        """视频播放主循环"""
        frame_delay = 1.0 / fps
        self.is_playing = True
        self.current_frame = 0

        # 创建状态显示线程
        status_thread = threading.Thread(target=self.show_status)
        status_thread.daemon = True
        status_thread.start()

        while self.is_playing and self.current_frame < self.total_frames:
            if self.pause_flag:
                time.sleep(0.1)
                continue

            start_time = time.time()

            try:
                # 加载当前帧
                if self.current_frame < len(preloaded_frames):
                    # 使用预加载的数据
                    points, colors = preloaded_frames[self.current_frame]
                else:
                    # 动态加载
                    file_path = files[self.current_frame]
                    points_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                    points = points_data[:, :3]
                    colors = self.color_by_elevation(points)

                # 更新点云
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)

                # 更新可视化
                self.vis.update_geometry(self.pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

                # 处理键盘输入
                if self.handle_key_events():
                    break

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                time.sleep(sleep_time)

                self.current_frame += 1

            except Exception as e:
                print(f"播放帧 {self.current_frame} 时出错: {e}")
                self.current_frame += 1
                continue

        self.is_playing = False
        print("\n播放结束")

    def handle_key_events(self):
        """处理键盘事件"""
        # Open3D的键盘事件处理有限，我们使用简单的状态检查
        # 在实际应用中，可能需要使用其他方法处理键盘输入
        return False  # 暂时返回False，不处理特殊按键

    def show_status(self):
        """显示播放状态"""
        while self.is_playing:
            progress = (self.current_frame / self.total_frames) * 100
            status = "暂停" if self.pause_flag else "播放中"
            print(f"\r{status}: 帧 {self.current_frame}/{self.total_frames} ({progress:.1f}%)", end="", flush=True)
            time.sleep(0.1)

    def color_by_elevation(self, points):
        """基于高程着色 - 创建更自然的视觉效果"""
        colors = np.zeros((len(points), 3))

        if len(points) == 0:
            return colors

        # 计算高程（Z坐标）
        z = points[:, 2]
        z_min, z_max = np.min(z), np.max(z)

        if z_max > z_min:
            # 归一化高程
            normalized_z = (z - z_min) / (z_max - z_min)

            # 创建自然色带：低处深色，高处亮色
            # 深蓝 -> 绿色 -> 黄色 -> 白色
            for i, z_val in enumerate(normalized_z):
                if z_val < 0.2:
                    # 深蓝到蓝绿
                    colors[i] = [0, 0, 0.2 + z_val * 2.5]
                elif z_val < 0.5:
                    # 蓝绿到绿色
                    colors[i] = [0, (z_val - 0.2) * 3.33, 0.7 - (z_val - 0.2) * 1.4]
                elif z_val < 0.8:
                    # 绿色到黄色
                    colors[i] = [(z_val - 0.5) * 3.33, 1, 0]
                else:
                    # 黄色到白色
                    colors[i] = [1, 1, (z_val - 0.8) * 5]
        else:
            # 所有点高度相同，使用蓝色
            colors[:, 2] = 1.0

        return colors

    def toggle_pause(self):
        """切换暂停状态"""
        self.pause_flag = not self.pause_flag

    def stop(self):
        """停止播放"""
        self.is_playing = False

def create_advanced_video_player():
    """创建高级视频播放器，支持更多功能"""
    player = PointCloudVideoPlayer()

    # 播放序列00的前500帧，10FPS
    player.create_smooth_video(
        sequence_id="00",
        frame_range=(0, 500),  # 前500帧
        fps=10,               # 10帧/秒
        point_size=1.0,       # 点大小
        background_color=[0.05, 0.05, 0.1]  # 深蓝色背景
    )

def benchmark_playback(sequence_id="00", num_frames=100):
    """测试播放性能"""
    print("=== 播放性能测试 ===")

    source_dir = Path(f"../data/raw_dataset/dataset/sequences/{sequence_id}/velodyne")
    files = sorted(source_dir.glob("*.bin"))[:num_frames]

    if not files:
        print("没有找到文件")
        return

    # 测试加载速度
    load_times = []
    for i, file_path in enumerate(files):
        start_time = time.time()
        points_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        load_time = time.time() - start_time
        load_times.append(load_time)

        if i % 20 == 0:
            print(f"测试加载帧 {i}/{num_frames}")

    avg_load_time = np.mean(load_times)
    max_load_time = np.max(load_times)

    print(f"平均加载时间: {avg_load_time*1000:.2f}ms")
    print(f"最大加载时间: {max_load_time*1000:.2f}ms")
    print(f"建议最大FPS: {1/max_load_time:.1f}")

def create_custom_trajectory_video(sequence_id="00", frame_range=None, trajectory_type="circle"):
    """创建带有自定义相机轨迹的视频"""
    source_dir = Path(f"../data/raw_dataset/dataset/sequences/{sequence_id}/velodyne")
    files = sorted(source_dir.glob("*.bin"))

    if frame_range:
        files = files[frame_range[0]:frame_range[1]]

    total_frames = len(files)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"点云视频 - 相机轨迹 - {trajectory_type}",
        width=1400,
        height=900
    )

    pcd = o3d.geometry.PointCloud()

    # 加载第一帧
    if files:
        points_data = np.fromfile(files[0], dtype=np.float32).reshape(-1, 4)
        points = points_data[:, :3]
        colors = color_by_distance(points)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.2])
    render_option.point_size = 1.2
    render_option.light_on = True

    # 获取视角控制器
    view_control = vis.get_view_control()

    print(f"开始播放带有 {trajectory_type} 相机轨迹的视频...")

    for frame_idx in range(total_frames):
        # 加载点云
        points_data = np.fromfile(files[frame_idx], dtype=np.float32).reshape(-1, 4)
        points = points_data[:, :3]
        colors = color_by_distance(points)

        # 更新点云
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 更新相机位置（创建轨迹效果）
        if trajectory_type == "circle":
            # 圆形轨迹
            angle = (frame_idx / total_frames) * 2 * np.pi
            radius = 50
            camera_pos = [
                radius * np.cos(angle),
                radius * np.sin(angle) * 0.5,
                10 + 5 * np.sin(angle * 2)
            ]
            view_control.set_lookat([0, 0, 0])
            view_control.set_front([-camera_pos[0], -camera_pos[1], -camera_pos[2]])
            view_control.set_up([0, 0, 1])

        elif trajectory_type == "follow":
            # 跟随车辆轨迹
            if len(points) > 0:
                # 假设车辆在原点附近，计算点云中心
                center = np.mean(points, axis=0)
                view_control.set_lookat(center)

                # 相机位置在车辆后方上方
                camera_distance = 20
                camera_height = 5
                view_control.set_front([0, -1, -0.2])
                view_control.set_up([0, -0.2, 1])

        # 更新可视化
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 显示进度
        progress = (frame_idx / total_frames) * 100
        print(f"\r播放进度: {frame_idx}/{total_frames} ({progress:.1f}%)", end="")

        # 控制帧率
        time.sleep(0.05)  # 约20FPS

    print("\n播放完成!")
    vis.destroy_window()

def color_by_distance(points):
    """基于距离着色"""
    colors = np.zeros((len(points), 3))

    if len(points) == 0:
        return colors

    # 计算每个点到原点的距离
    distances = np.sqrt(np.sum(points**2, axis=1))
    dist_min, dist_max = np.min(distances), np.max(distances)

    if dist_max > dist_min:
        normalized_dist = (distances - dist_min) / (dist_max - dist_min)

        # 使用热力图：近处蓝色，远处红色
        colors[:, 0] = normalized_dist  # 红色
        colors[:, 2] = 1 - normalized_dist  # 蓝色
        colors[:, 1] = 0.3  # 少量绿色
    else:
        colors[:, :] = [0.7, 0.7, 0.7]  # 灰色

    return colors

def play_multiple_sequences():
    """播放多个序列进行比较"""
    sequences = [
        ("00", "城市街道"),
        ("01", "高速公路"),
        ("02", "乡村道路"),
        ("03", "居民区")
    ]

    for seq_id, description in sequences:
        source_dir = Path(f"../data/raw_dataset/dataset/sequences/{seq_id}/velodyne")
        if not source_dir.exists():
            continue

        files = sorted(source_dir.glob("*.bin"))
        if not files:
            continue

        print(f"\n=== 播放序列 {seq_id}: {description} ===")
        print(f"总帧数: {len(files)}")

        player = PointCloudVideoPlayer()
        player.create_smooth_video(
            sequence_id=seq_id,
            frame_range=(0, min(200, len(files))),  # 最多播放200帧
            fps=10,
            point_size=1.0,
            background_color=[0.1, 0.1, 0.15]
        )

        # 询问是否继续下一个序列
        if seq_id != sequences[-1][0]:
            response = input("继续播放下一个序列? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == "__main__":
    print("=== 点云视频播放器 ===")
    print("1. 流畅播放序列00")
    print("2. 带相机轨迹的播放")
    print("3. 性能测试")
    print("4. 播放多个序列")
    print("5. 退出")

    choice = input("请选择: ").strip()

    if choice == "1":
        create_advanced_video_player()
    elif choice == "2":
        print("选择相机轨迹:")
        print("  1. 圆形轨迹")
        print("  2. 跟随轨迹")
        traj_choice = input("请选择: ").strip()

        if traj_choice == "1":
            create_custom_trajectory_video("00", (0, 300), "circle")
        else:
            create_custom_trajectory_video("00", (0, 300), "follow")
    elif choice == "3":
        benchmark_playback("00", 100)
    elif choice == "4":
        play_multiple_sequences()
    elif choice == "5":
        print("退出")
    else:
        print("无效选择")