import os
import numpy as np
import open3d as o3d
import argparse
import time
import json
import math
from pathlib import Path
from functools import partial

# --- S3DIS 标签 -> 中文名称映射 ---
S3DIS_LABEL_MAP = {
    0: '天花板 (ceiling)',
    1: '地板 (floor)',
    2: '墙 (wall)',
    3: '横梁 (beam)',
    4: '柱子 (column)',
    5: '窗户 (window)',
    6: '门 (door)',
    7: '桌子 (table)',
    8: '椅子 (chair)',
    9: '沙发 (sofa)',
    10: '书柜 (bookcase)',
    11: '白板 (board)',
    12: '杂物 (clutter)'
}
NUM_CLASSES = len(S3DIS_LABEL_MAP)

# --- 实例ID到语义类别的映射函数 ---
def map_instance_to_semantic(instance_labels, instance2class_file=None):
    if instance2class_file and os.path.exists(instance2class_file):
        instance2class = {}
        with open(instance2class_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    instance_id = int(parts[0])
                    class_id = int(parts[1])
                    instance2class[instance_id] = class_id
        
        semantic_labels = np.zeros_like(instance_labels)
        for instance_id, class_id in instance2class.items():
            mask = (instance_labels == instance_id).flatten()
            semantic_labels[mask] = class_id
        return semantic_labels
    else:
        print("警告: 未找到实例到类别的映射文件，使用自动映射")
        semantic_labels = instance_labels % NUM_CLASSES
        return semantic_labels

# --- 为实例分割生成独特颜色的函数 ---
def get_instance_colors(instance_labels):
    if instance_labels.size == 0:
        return np.empty((0, 3))
    
    if instance_labels.ndim > 1:
        instance_labels = instance_labels.flatten()
    
    instance_labels = instance_labels.astype(int)
    unique_instances = np.unique(instance_labels)
    colors = np.zeros((instance_labels.shape[0], 3))
    
    for instance_id in unique_instances:
        hue = (instance_id * 0.618033988749895) % 1.0
        saturation = 0.7 + (instance_id % 4) * 0.1
        value = 0.7 + (instance_id % 3) * 0.15
        
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)
        
        if h_i == 0:
            rgb = np.array([value, t, p])
        elif h_i == 1:
            rgb = np.array([q, value, p])
        elif h_i == 2:
            rgb = np.array([p, value, t])
        elif h_i == 3:
            rgb = np.array([p, q, value])
        elif h_i == 4:
            rgb = np.array([t, p, value])
        else:
            rgb = np.array([value, p, q])
        
        mask = (instance_labels == instance_id)
        if mask.ndim > 1:
            mask = mask.flatten()
        colors[mask] = rgb
    
    return colors

# --- 修改后的 get_label_colors 函数 ---
def get_label_colors(labels, label_type='semantic', instance2class_file=None):
    if labels.size == 0:
        return np.empty((0, 3))

    if label_type == 'instance':
        return get_instance_colors(labels)

    s3dis_colors = np.array([
        [152, 223, 138],   # 0 ceiling
        [174, 199, 232],   # 1 floor
        [255, 187, 120],   # 2 wall
        [255, 152, 150],   # 3 beam
        [197, 176, 213],   # 4 column
        [196, 156, 148],   # 5 window
        [247, 182, 210],   # 6 door
        [199, 199, 199],   # 7 table
        [219, 219, 141],   # 8 chair
        [158, 218, 229],   # 9 sofa
        [188, 189, 34],    # 10 bookcase
        [255, 127, 14],    # 11 board
        [140, 86, 75]      # 12 clutter
    ]) / 255.0

    if labels.ndim > 1:
        labels = labels.flatten()
    
    labels_int = labels.astype(int)
    
    max_label = s3dis_colors.shape[0] - 1
    if np.any(labels_int > max_label):
        print(f"警告: 标签中存在超出S3DIS官方定义的类别ID (最大为 {max_label})。")
        default_color = np.array([0.5, 0.5, 0.5])
        colors = np.full((labels_int.shape[0], 3), default_color)
        valid_mask = labels_int <= max_label
        colors[valid_mask] = s3dis_colors[labels_int[valid_mask]]
    else:
        colors = s3dis_colors[labels_int]
    
    return colors

# --- 打印颜色标签图例的函数 ---
def print_color_legend(labels, label_type='semantic', instance2class_file=None):
    if labels.size == 0:
        return
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    if label_type == 'instance':
        unique_instances = np.unique(labels)
        print(f"\n--- 实例分割图例 ---")
        print(f"实例统计: 共 {len(unique_instances)} 个实例对象")
        print("每个实例都有独特的颜色")
        
        if instance2class_file and os.path.exists(instance2class_file):
            print("\n前10个实例的语义类别映射:")
            for instance_id in sorted(unique_instances)[:10]:
                semantic_id = map_instance_to_semantic(np.array([instance_id]), instance2class_file)[0]
                semantic_name = S3DIS_LABEL_MAP.get(semantic_id, f"未知ID {semantic_id}")
                print(f"  实例 {instance_id} -> 语义类别 {semantic_id} ({semantic_name})")
        else:
            print("\n前10个实例:")
            for instance_id in sorted(unique_instances)[:10]:
                print(f"  实例 {instance_id}")
                
        if len(unique_instances) > 10:
            print(f"  ... 和其他 {len(unique_instances) - 10} 个实例")
    else:
        unique_semantic = np.unique(labels)
        title = f"{label_type.capitalize()}颜色图例"
        
        print(f"\n--- {title} ---")
        for label_id in sorted(unique_semantic):
            color_for_label = get_label_colors(np.array([label_id]), label_type='semantic')[0]
            rgb_255 = (color_for_label * 255).astype(int)
            color_block = f"\033[48;2;{rgb_255[0]};{rgb_255[1]};{rgb_255[2]}m  \033[0m"
            label_text = S3DIS_LABEL_MAP.get(label_id, f"未知标签ID: {label_id}")
            print(f" {color_block} Label {label_id:<2} ({label_text})")
    
    print("----------------------")

# ===================== 相机辅助函数（修正版）=====================
def get_view_angles(vis, center=None):
    """获取当前窗口的方位角、仰角、roll和相机位置（用于保存视角）
    center: 点云的中心点，如果提供，则计算相机到中心点的距离
    """
    ctrl = vis.get_view_control()
    # 获取相机参数
    params = ctrl.convert_to_pinhole_camera_parameters()
    extrinsic = params.extrinsic  # 4x4 矩阵，相机到世界的变换
    
    # 计算相机位置
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam_pos = -R.T @ t  # 世界坐标系中的相机位置
    
    # 计算前向向量（从相机指向场景前方）
    # 相机的Z轴指向相机后方，所以前向是负Z轴
    front = -R[:, 2]
    front = front / np.linalg.norm(front)
    
    # 计算上方向量
    up = R[:, 1]
    up = up / np.linalg.norm(up)
    
    # 计算右方向量
    right = R[:, 0]
    right = right / np.linalg.norm(right)
    
    # 计算注视点
    if center is not None:
        # 使用点云中心点作为注视点
        lookat = center
    else:
        # 如果没有中心点信息，使用相机前方1单位处作为注视点
        lookat = cam_pos + front
    
    # 计算距离
    distance = np.linalg.norm(cam_pos - lookat)
    
    # 计算方位角、仰角、roll
    # 方位角：从Z轴正方向到相机在XZ平面上的投影的角度
    azimuth = math.atan2(front[0], front[2])
    # 仰角：相机在前向方向与XZ平面的夹角
    elevation = math.asin(front[1])
    # 计算 roll：上向量绕前向轴的旋转角
    # 理想上方向：垂直于前向向量，且在包含前向向量和全球Z轴的平面内
    global_up = np.array([0.0, 0.0, 1.0])  # 全球坐标系的Z轴（S3DIS数据集使用Z轴向上）
    # 计算理想上方向
    ideal_right = np.cross(global_up, front)
    if np.linalg.norm(ideal_right) < 1e-6:
        ideal_right = np.array([1.0, 0.0, 0.0])
    ideal_right = ideal_right / np.linalg.norm(ideal_right)
    ideal_up = np.cross(front, ideal_right)
    ideal_up = ideal_up / np.linalg.norm(ideal_up)
    # 计算roll角
    roll = np.arctan2(np.dot(up, ideal_right), np.dot(up, ideal_up))
    
    return {
        'position': cam_pos,
        'lookat': lookat,
        'front': front,
        'up': up,
        'right': right,
        'distance': distance,
        'azimuth': math.degrees(azimuth),
        'elevation': math.degrees(elevation),
        'roll': math.degrees(roll)
    }

def set_view_angles(vis, center, azimuth_deg, elevation_deg, roll_deg, distance):
    """
    根据方位角、仰角、roll（度）和场景中心设置视角。
    distance: 相机到场景中心的距离。
    """
    ctrl = vis.get_view_control()
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)
    roll_rad = math.radians(roll_deg)
    
    # 球坐标转笛卡尔坐标（相机位置）
    # 方位角：从Z轴正方向到相机在XZ平面上的投影的角度
    # 仰角：相机在前向方向与XZ平面的夹角
    # 与get_view_angles函数保持一致的坐标系定义
    x = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = distance * math.sin(elevation_rad)
    z = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    cam_pos = center + np.array([x, y, z])
    
    # 计算前向向量（从相机指向中心）
    front = center - cam_pos
    front = front / np.linalg.norm(front)
    
    # 计算右向量和上向量
    # 全球坐标系的Z轴向上
    global_up = np.array([0.0, 0.0, 1.0])
    
    # 计算右向量
    right = np.cross(global_up, front)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    
    # 计算理想上方向（无roll）
    ideal_up = np.cross(front, right)
    ideal_up = ideal_up / np.linalg.norm(ideal_up)
    
    # 应用roll旋转（绕前向轴）
    # 注意：这里的roll旋转方向应该与get_view_angles函数中的计算保持一致
    up = ideal_up * math.cos(roll_rad) - right * math.sin(roll_rad)
    up = up / np.linalg.norm(up)
    
    # 重新计算右向量，确保三个向量正交
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    
    # 设置相机参数
    # Open3D的set_front方法期望的是从注视点指向相机的方向
    ctrl.set_lookat(center)
    ctrl.set_front(-front)  # 从注视点指向相机的方向
    ctrl.set_up(up)

# ===================== 相机同步类（修正版）=====================
class CameraSynchronizer:
    """管理多窗口相机参数的保存、加载和同步"""
    def __init__(self, visualizers, initial_params=None):
        self.visualizers = visualizers
        self.saved_params = None
        self.initial_params = initial_params  # (center, distance, azimuth, elevation, roll)
        self.center = initial_params[0] if initial_params else None  # 点云中心点

    def get_camera_params(self, vis):
        ctrl = vis.get_view_control()
        params = ctrl.convert_to_pinhole_camera_parameters()
        return {
            'extrinsic': params.extrinsic.tolist(),
            'intrinsic': {
                'width': params.intrinsic.width,
                'height': params.intrinsic.height,
                'fx': params.intrinsic.intrinsic_matrix[0,0],
                'fy': params.intrinsic.intrinsic_matrix[1,1],
                'cx': params.intrinsic.intrinsic_matrix[0,2],
                'cy': params.intrinsic.intrinsic_matrix[1,2]
            }
        }

    def set_camera_params(self, vis, params):
        """只恢复外参（相机位姿），内参保持当前窗口的"""
        ctrl = vis.get_view_control()
        # 获取当前窗口的内参
        current_params = ctrl.convert_to_pinhole_camera_parameters()
        current_intrinsic = current_params.intrinsic
        # 构建新的相机参数对象
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = current_intrinsic
        camera_params.extrinsic = np.array(params['extrinsic'])
        # 应用相机参数
        # 忽略窗口尺寸不匹配的警告
        try:
            ctrl.convert_from_pinhole_camera_parameters(camera_params, True)
        except Exception as e:
            # 即使有错误，也继续执行
            pass
        # 确保注视点是点云的中心点
        if self.center is not None:
            ctrl.set_lookat(self.center)

    def print_camera(self, idx):
        params = self.get_camera_params(self.visualizers[idx])
        print(f"\n--- Window {idx} Camera Parameters ---")
        print("Extrinsic matrix (4x4):")
        print(np.array(params['extrinsic']))
        print("Intrinsic parameters:")
        print(f"  width={params['intrinsic']['width']}, height={params['intrinsic']['height']}")
        print(f"  fx={params['intrinsic']['fx']}, fy={params['intrinsic']['fy']}")
        print(f"  cx={params['intrinsic']['cx']}, cy={params['intrinsic']['cy']}")
        print("------------------------------------\n")

    def save_camera(self, idx):
        self.saved_params = self.get_camera_params(self.visualizers[idx])
        print(f"Camera parameters saved from window {idx}.")

    def load_camera(self, idx):
        if self.saved_params is None:
            print("No saved camera parameters. Press 'S' to save first.")
            return
        self.set_camera_params(self.visualizers[idx], self.saved_params)
        print(f"Loaded saved camera parameters to window {idx}.")

    def apply_to_all(self):
        if self.saved_params is None:
            print("No saved camera parameters. Press 'S' to save first.")
            return
        # 打印同步前的视角角度
        print("\n--- 同步前视角角度 ---")
        for idx, vis in enumerate(self.visualizers):
            view = get_view_angles(vis, self.center)
            print(f"窗口 {idx}: 方位角={view['azimuth']:.1f}°, 仰角={view['elevation']:.1f}°, roll={view['roll']:.1f}°")
        # 应用保存的相机参数
        for idx, vis in enumerate(self.visualizers):
            self.set_camera_params(vis, self.saved_params)
        # 打印同步后的视角角度
        print("\n--- 同步后视角角度 ---")
        for idx, vis in enumerate(self.visualizers):
            view = get_view_angles(vis, self.center)
            print(f"窗口 {idx}: 方位角={view['azimuth']:.1f}°, 仰角={view['elevation']:.1f}°, roll={view['roll']:.1f}°")
        print("Applied saved camera parameters to all windows.")

    def reset_view(self, idx):
        if self.initial_params is None:
            print("No initial view parameters available.")
            return
        center, distance, azimuth, elevation, roll = self.initial_params
        set_view_angles(self.visualizers[idx], center, azimuth, elevation, roll, distance)
        print(f"Reset view to initial azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, roll={roll:.1f}°")

    def save_view(self, idx):
        view = get_view_angles(self.visualizers[idx], self.center)
        print(f"\n[保存视角] 相机位置: {view['position']}")
        print(f"[保存视角] 方位角: {view['azimuth']:.1f}°, 仰角: {view['elevation']:.1f}°, roll: {view['roll']:.1f}°")
        print(f"[使用提示] 要重现此视角，请使用 --view_angle={view['azimuth']:.1f},{view['elevation']:.1f},{view['roll']:.1f} 参数")

    def print_view(self, idx):
        view = get_view_angles(self.visualizers[idx], self.center)
        print(f"\n[当前视角] 相机位置: {view['position']}")
        print(f"[当前视角] 方位角: {view['azimuth']:.1f}°, 仰角: {view['elevation']:.1f}°, roll: {view['roll']:.1f}°")
        print(f"[使用提示] 要重现此视角，请使用 --view_angle={view['azimuth']:.1f},{view['elevation']:.1f},{view['roll']:.1f} 参数")

# ===================== 辅助函数：创建不返回值的回调 =====================
def make_callback(func, *args):
    """返回一个接受vis参数的函数，调用func(*args)并更新渲染"""
    def callback(vis):
        func(*args)
        vis.update_renderer()
    return callback

# ===================== 可视化核心函数 =====================
def visualize_flexible_pointclouds(coords, window_configs, scene_id, args):
    try:
        print(f"坐标形状: {coords.shape}")
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            print(f"窗口 {i+1} ({window_type}) 颜色形状: {colors.shape}")
        
        # 调整颜色数组形状
        def ensure_correct_shape(colors, expected_points, name):
            if colors.ndim == 3:
                colors = colors.reshape(-1, 3)
            if colors.shape[0] != expected_points:
                print(f"警告: {name} 数据点数量 ({colors.shape[0]}) 与坐标数据点数量 ({expected_points}) 不匹配")
                if colors.shape[0] < expected_points:
                    repeat_times = expected_points // colors.shape[0] + 1
                    colors = np.tile(colors, (repeat_times, 1))[:expected_points]
                else:
                    colors = colors[:expected_points]
            return colors
        
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            window_configs[i] = (window_type,
                               ensure_correct_shape(colors, coords.shape[0], f"{window_name}颜色"),
                               window_name)
        
        # 创建点云和可视化器（使用 VisualizerWithKeyCallback 以支持键盘回调）
        pcds = []
        visualizers = []
        for window_type, colors, window_name in window_configs:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pcd)
            vis = o3d.visualization.VisualizerWithKeyCallback()
            visualizers.append(vis)
        
        # 设置窗口布局
        window_width = args.win_width
        window_height = args.win_height
        num_windows = len(window_configs)
        if num_windows > 4:
            window_width = min(window_width, 400)
            window_height = min(window_height, 350)
        elif num_windows > 2:
            window_width = min(window_width, 450)
            window_height = min(window_height, 400)
        
        screen_width = 1920
        screen_height = 1080
        if num_windows <= 2:
            cols = num_windows
            rows = 1
        elif num_windows <= 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = (num_windows + cols - 1) // cols
        
        horizontal_gap = 20
        vertical_gap = 40
        start_x = (screen_width - (cols * window_width + (cols - 1) * horizontal_gap)) // 2
        start_y = 50
        
        # 创建窗口并添加点云
        for i, ((window_type, colors, window_name), vis) in enumerate(zip(window_configs, visualizers)):
            row = i // cols
            col = i % cols
            x_pos = start_x + col * (window_width + horizontal_gap)
            y_pos = start_y + row * (window_height + vertical_gap)
            vis.create_window(
                window_name=f"{window_name} - {scene_id}",
                width=window_width,
                height=window_height,
                left=x_pos,
                top=y_pos
            )
            vis.add_geometry(pcds[i])
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([1.0, 1.0, 1.0])
        
        # 计算场景中心
        center = coords.mean(axis=0)
        
        # 获取第一个窗口的默认相机参数，用于计算初始距离
        for vis in visualizers:
            vis.poll_events()
            vis.update_renderer()
        ctrl0 = visualizers[0].get_view_control()
        params0 = ctrl0.convert_to_pinhole_camera_parameters()
        extrinsic0 = params0.extrinsic
        R0 = extrinsic0[:3, :3]
        t0 = extrinsic0[:3, 3]
        default_cam_pos = -R0.T @ t0
        default_distance = np.linalg.norm(default_cam_pos - center)
        
        # 解析初始视角参数
        initial_azimuth = None
        initial_elevation = None
        initial_roll = None
        if args.view_angle:
            try:
                parts = args.view_angle.split(',')
                if len(parts) >= 2:
                    initial_azimuth = float(parts[0])
                    initial_elevation = float(parts[1])
                    initial_roll = float(parts[2]) if len(parts) > 2 else 0.0
            except:
                print("警告: --view_angle 解析失败，使用默认视角")
        
        if initial_azimuth is not None:
            for vis in visualizers:
                set_view_angles(vis, center, initial_azimuth, initial_elevation, initial_roll, default_distance)
            view0 = get_view_angles(visualizers[0], center)
            initial_azimuth = view0['azimuth']
            initial_elevation = view0['elevation']
            initial_roll = view0['roll']
        else:
            view0 = get_view_angles(visualizers[0], center)
            initial_azimuth = view0['azimuth']
            initial_elevation = view0['elevation']
            initial_roll = view0['roll']
        
        initial_params = (center, default_distance, initial_azimuth, initial_elevation, initial_roll)
        
        # 相机同步器
        sync = CameraSynchronizer(visualizers, initial_params)
        
        # 为每个窗口注册键盘回调（使用 make_callback 确保返回 None）
        for idx, vis in enumerate(visualizers):
            # 大写 P：打印完整相机参数
            vis.register_key_callback(80, make_callback(sync.print_camera, idx))
            # 大写 S：保存相机参数到内存
            vis.register_key_callback(83, make_callback(sync.save_camera, idx))
            # 大写 L：加载保存的相机参数
            vis.register_key_callback(76, make_callback(sync.load_camera, idx))
            # 大写 A：同步所有窗口
            vis.register_key_callback(65, make_callback(sync.apply_to_all))
            # 小写 s：保存当前视角（输出可重现参数）
            vis.register_key_callback(115, make_callback(sync.save_view, idx))
            # 小写 r：重置视角
            vis.register_key_callback(114, make_callback(sync.reset_view, idx))
            # 小写 v：打印当前视角信息
            vis.register_key_callback(118, make_callback(sync.print_view, idx))
        
        # 加载相机参数文件（可选）
        if args.camera_file and os.path.exists(args.camera_file):
            try:
                with open(args.camera_file, 'r') as f:
                    loaded_params = json.load(f)
                sync.saved_params = loaded_params
                sync.apply_to_all()
                print(f"Loaded camera parameters from {args.camera_file}")
            except Exception as e:
                print(f"警告: 加载相机参数文件失败: {e}")
        
        # 显示信息
        print(f"显示 {num_windows} 个窗口对比: {scene_id}")
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            print(f"  窗口 {i+1}: {window_name}")
        print("键盘快捷键:")
        print("  P = 打印完整相机参数")
        print("  S = 保存相机参数到内存")
        print("  L = 加载保存的相机参数")
        print("  A = 同步所有窗口")
        print("  s = 保存当前视角（输出可重现参数）")
        print("  r = 重置视角")
        print("  v = 打印当前视角信息")
        print("关闭任意窗口继续...")
        
        # 运行可视化
        try:
            while True:
                all_active = True
                for vis in visualizers:
                    if not vis.poll_events():
                        all_active = False
                    vis.update_renderer()
                if not all_active:
                    break
                time.sleep(0.01)
        finally:
            # 打印最终视角
            if visualizers:
                final_view = get_view_angles(visualizers[0])
                print(f"\n[最终视角] 相机位置: {final_view['position']}")
                print(f"[最终视角] 方位角: {final_view['azimuth']:.1f}°, 仰角: {final_view['elevation']:.1f}°, roll: {final_view['roll']:.1f}°")
                print(f"[使用提示] 要重现此视角，请使用 --view_angle={final_view['azimuth']:.1f},{final_view['elevation']:.1f},{final_view['roll']:.1f} 参数")
            # 保存相机参数
            if args.save_camera and visualizers:
                params = sync.get_camera_params(visualizers[0])
                with open(args.save_camera, 'w') as f:
                    json.dump(params, f, indent=2)
                print(f"Saved camera parameters to {args.save_camera}")
            for vis in visualizers:
                vis.destroy_window()
    except Exception as e:
        import traceback
        print(f"在多窗口可视化过程中出错: {str(e)}")
        traceback.print_exc()
        raise

# --- 单窗口可视化函数（无键盘回调）---
def visualize_single_pointcloud(coords, colors, title="", output_path=None, delay=0, win_width=1280, win_height=720):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if output_path:
        width, height = 1920, 1080
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        render.scene.add_geometry("pointcloud", pcd, material)
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        size = np.linalg.norm(bbox.get_extent())
        direction = np.array([0.5, -1.0, 0.8])
        normalized_direction = direction / np.linalg.norm(direction)
        distance_multiplier = 0.7
        distance = size * distance_multiplier
        eye = center + normalized_direction * distance
        up = np.array([0, 0, 1])
        render.scene.camera.look_at(center, eye, up)
        img = render.render_to_image()
        o3d.io.write_image(output_path, img, 9)
        print(f"Saved visualization to {output_path}")
    else:
        print(f"Displaying: {title}. Close the window to continue.")
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window(window_name=title, width=win_width, height=win_height)
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([1.0, 1.0, 1.0])
            vis.run()
        finally:
            vis.destroy_window()

        if delay > 0:
            time.sleep(delay)

# --- 指标计算函数 ---
def calculate_and_print_metrics(pred_labels, gt_labels, pred_name="预测"):
    if pred_labels.ndim > 1: pred_labels = pred_labels.flatten()
    if gt_labels.ndim > 1: gt_labels = gt_labels.flatten()

    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(conf_matrix, (gt_labels.astype(int), pred_labels.astype(int)), 1)

    intersection = np.diag(conf_matrix)
    gt_per_class = conf_matrix.sum(axis=1)
    pred_per_class = conf_matrix.sum(axis=0)
    union = gt_per_class + pred_per_class - intersection
    
    iou = intersection / (union + np.finfo(float).eps)
    class_acc = intersection / (gt_per_class + np.finfo(float).eps)

    valid_classes = gt_per_class > 0
    mean_iou = np.mean(iou[valid_classes])
    overall_acc = intersection.sum() / (gt_per_class.sum() + np.finfo(float).eps)

    print(f"\n--- [{pred_name} 评估指标] ---")
    print(f"{'类别名称':<20} {'IoU':>8} {'Accuracy':>10}")
    print("-" * 40)
    for i in range(NUM_CLASSES):
        label_name = S3DIS_LABEL_MAP.get(i, f"未知ID {i}")
        if valid_classes[i]:
            print(f"{label_name:<20} {iou[i]:>8.4f} {class_acc[i]:>10.4f}")
    print("-" * 40)
    print(f"{'Overall Accuracy (allAcc)':<30} {overall_acc:>8.4f}")
    print(f"{'Mean IoU (mIoU)':<30} {mean_iou:>8.4f}")
    print("---------------------\n")

# --- 核心处理函数 ---
def process_and_visualize_flexible(coords_path, args, scene_id,
                                  pred1_path=None, pred2_path=None,
                                  gt_path=None, color_path=None,
                                  instance_path=None):
    try:
        coords = np.load(coords_path)
        window_configs = []
        
        # 解析 solid_color
        try:
            solid_color = [float(x) for x in args.solid_color.split(',')]
            if max(solid_color) > 1.0:
                solid_color = [c / 255.0 for c in solid_color]
            if len(solid_color) != 3:
                raise ValueError
        except:
            print(f"警告：无效的 solid_color 参数 '{args.solid_color}'，使用默认灰色 [0.5,0.5,0.5]")
            solid_color = [0.5, 0.5, 0.5]
        
        window_types = args.windows.split(',') if args.windows else []
        for window_type in window_types:
            window_type = window_type.strip().lower()
            if window_type == 'color':
                if color_path is None:
                    color_path = os.path.join(os.path.dirname(coords_path), "color.npy")
                if os.path.exists(color_path):
                    print(f"加载颜色文件: {color_path}")
                    colors = np.load(color_path)
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    window_configs.append(('color', colors, "原始颜色"))
                else:
                    print(f"警告: 颜色文件不存在: {color_path}，跳过此窗口")
            elif window_type == 'gt':
                if gt_path and os.path.exists(gt_path):
                    print(f"加载真值文件: {gt_path}")
                    gt_labels = np.load(gt_path)
                    gt_colors = get_label_colors(gt_labels, label_type='semantic')
                    window_configs.append(('gt', gt_colors, "真值语义"))
                    print_color_legend(gt_labels, "semantic")
                else:
                    print(f"警告: 真值文件不存在: {gt_path}，跳过此窗口")
            elif window_type == 'instance':
                if instance_path is None:
                    instance_path = os.path.join(os.path.dirname(coords_path), "instance.npy")
                if os.path.exists(instance_path):
                    print(f"加载实例文件: {instance_path}")
                    instance_labels = np.load(instance_path)
                    instance2class_file = None
                    data_dir = os.path.dirname(coords_path)
                    potential_files = [
                        os.path.join(data_dir, "instance2class.txt"),
                        os.path.join(data_dir, "instances2classes.txt"),
                        os.path.join(data_dir, "instance_to_class.txt")
                    ]
                    for file_path in potential_files:
                        if os.path.exists(file_path):
                            instance2class_file = file_path
                            print(f"找到实例映射文件: {instance2class_file}")
                            break
                    instance_colors = get_label_colors(instance_labels, label_type='instance', instance2class_file=instance2class_file)
                    window_configs.append(('instance', instance_colors, "实例分割"))
                    print_color_legend(instance_labels, "instance", instance2class_file)
                else:
                    print(f"警告: 实例文件不存在: {instance_path}，跳过此窗口")
            elif window_type == 'pred1':
                if pred1_path and os.path.exists(pred1_path):
                    print(f"加载预测1文件: {pred1_path}")
                    pred1_labels = np.load(pred1_path)
                    if pred1_labels.ndim == 2 and pred1_labels.shape[1] == 1:
                        pred1_labels = pred1_labels.flatten()
                    pred1_colors = get_label_colors(pred1_labels, label_type='semantic')
                    window_configs.append(('pred1', pred1_colors, args.pred1_name))
                    if gt_path and os.path.exists(gt_path):
                        gt_labels = np.load(gt_path)
                        calculate_and_print_metrics(pred1_labels, gt_labels, args.pred1_name)
                    print(f"\n--- {args.pred1_name} 图例 ---")
                    print_color_legend(pred1_labels, "semantic")
                else:
                    print(f"警告: 预测1文件不存在: {pred1_path}，跳过此窗口")
            elif window_type == 'pred2':
                if pred2_path and os.path.exists(pred2_path):
                    print(f"加载预测2文件: {pred2_path}")
                    pred2_labels = np.load(pred2_path)
                    if pred2_labels.ndim == 2 and pred2_labels.shape[1] == 1:
                        pred2_labels = pred2_labels.flatten()
                    pred2_colors = get_label_colors(pred2_labels, label_type='semantic')
                    window_configs.append(('pred2', pred2_colors, args.pred2_name))
                    if gt_path and os.path.exists(gt_path):
                        gt_labels = np.load(gt_path)
                        calculate_and_print_metrics(pred2_labels, gt_labels, args.pred2_name)
                    print(f"\n--- {args.pred2_name} 图例 ---")
                    print_color_legend(pred2_labels, "semantic")
                else:
                    print(f"警告: 预测2文件不存在: {pred2_path}，跳过此窗口")
            elif window_type == 'blank':
                colors = np.full((coords.shape[0], 3), solid_color)
                window_configs.append(('blank', colors, "纯色"))
            else:
                print(f"警告: 未知的窗口类型 '{window_type}'，跳过")
        
        if not window_configs:
            print("没有有效的窗口配置，使用默认单窗口显示")
            colors = np.full((coords.shape[0], 3), 0.5)
            title = f"默认视图 - {scene_id}"
            output_path = os.path.join(args.output_dir, f"{scene_id}_default.png") if args.output_dir else None
            visualize_single_pointcloud(coords, colors, title=title, output_path=output_path, delay=args.delay,
                                         win_width=args.win_width, win_height=args.win_height)
            return True
        
        print("开始多窗口可视化...")
        visualize_flexible_pointclouds(coords, window_configs, scene_id, args)
        return True
    except Exception as e:
        import traceback
        print(f"Error processing scene '{scene_id}': {str(e)}")
        traceback.print_exc()
        return False

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(
        description="灵活的点云可视化工具 (支持任意窗口组合，包括纯色窗口)。\n"
                    "新增视角控制功能：可通过 --view_angle 指定初始视角，\n"
                    "按 s 保存当前视角，按 r 重置视角，按 v 手动打印当前视角信息。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--coords", type=str, default=None, help="模式1: 单个点云坐标文件路径 (coord.npy)。")
    parser.add_argument("--pred", type=str, default=None, help="模式1: 与 --coords 配合使用，提供单个预测标签文件路径。")
    parser.add_argument("--pred_dir", type=str, default=None, help="模式2: 包含预测标签 (*_pred.npy) 的目录。")
    parser.add_argument("--data_root", type=str, default=None, help="模式2 & 3: 原始数据的根目录。")
    parser.add_argument("--windows", type=str, default="", help="指定要显示的窗口类型和顺序，用逗号分隔。")
    parser.add_argument("--compare_pred_dir", type=str, default=None, help="用于第二个预测的目录")
    parser.add_argument("--pred1_name", type=str, default="预测1", help="第一个预测的名称")
    parser.add_argument("--pred2_name", type=str, default="预测2", help="第二个预测的名称")
    parser.add_argument("--solid_color", type=str, default="0.5,0.5,0.5", help="纯色窗口的颜色")
    parser.add_argument("--camera_file", type=str, default=None, help="启动时加载的相机参数JSON文件")
    parser.add_argument("--save_camera", type=str, default=None, help="退出时保存相机参数到JSON文件")
    parser.add_argument("--view_angle", type=str, default=None, help="初始视角，格式 azimuth,elevation,roll")
    parser.add_argument("--mode", type=str, default="segment", choices=["color", "segment", "instance"])
    parser.add_argument("--output_dir", type=str, default=None, help="保存可视化图像的目录")
    parser.add_argument("--continuous", action="store_true", help="自动连续显示所有点云")
    parser.add_argument("--delay", type=float, default=0.5, help="连续显示时的延迟")
    parser.add_argument("--win_width", type=int, default=500, help="窗口宽度")
    parser.add_argument("--win_height", type=int, default=500, help="窗口高度")
    parser.add_argument("--three_windows", action="store_true", help="向后兼容：color,gt,pred1")
    parser.add_argument("--four_windows", action="store_true", help="向后兼容：color,gt,pred1,pred2")
    
    args = parser.parse_args()
    
    if args.three_windows and not args.windows:
        args.windows = "color,gt,pred1"
        print("使用三窗口模式: color,gt,pred1")
    if args.four_windows and not args.windows:
        args.windows = "color,gt,pred1,pred2"
        print("使用四窗口模式: color,gt,pred1,pred2")
    
    if args.windows:
        num_windows = len(args.windows.split(','))
        if num_windows > 4:
            args.win_width = min(args.win_width, 400)
            args.win_height = min(args.win_height, 350)
        elif num_windows > 2:
            args.win_width = min(args.win_width, 450)
            args.win_height = min(args.win_height, 400)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"可视化图像将保存到: {args.output_dir}")
    
    if args.coords:
        if not os.path.exists(args.coords): 
            raise FileNotFoundError(f"坐标文件不存在: {args.coords}")
        scene_id = Path(args.coords).parent.name
        data_dir = Path(args.coords).parent
        gt_path = data_dir / "segment.npy"
        color_path = data_dir / "color.npy"
        instance_path = data_dir / "instance.npy"
        pred1_path = None
        pred2_path = None
        if args.pred:
            pred1_path = args.pred
        elif args.windows and any(w in args.windows for w in ['pred1', 'pred']):
            pred1_path = data_dir / "pred.npy"
            if not pred1_path.exists():
                print(f"警告: 默认预测文件不存在: {pred1_path}")
        if args.compare_pred_dir:
            if os.path.isfile(args.compare_pred_dir):
                pred2_path = args.compare_pred_dir
                print(f"使用指定的预测2文件: {pred2_path}")
            else:
                pred2_filename = f"{scene_id}_pred.npy"
                pred2_path = os.path.join(args.compare_pred_dir, pred2_filename)
                print(f"构建预测2文件路径: {pred2_path}")
            if not os.path.exists(pred2_path):
                print(f"警告: 预测2文件不存在: {pred2_path}，跳过此窗口")
        process_and_visualize_flexible(
            args.coords, args, scene_id,
            pred1_path=pred1_path,
            pred2_path=pred2_path,
            gt_path=str(gt_path) if gt_path.exists() else None,
            color_path=str(color_path) if color_path.exists() else None,
            instance_path=str(instance_path) if instance_path.exists() else None
        )
    elif args.pred_dir and args.data_root:
        if not os.path.isdir(args.pred_dir): 
            raise FileNotFoundError(f"预测文件目录不存在: {args.pred_dir}")
        if not os.path.isdir(args.data_root): 
            raise FileNotFoundError(f"原始数据根目录不存在: {args.data_root}")
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith("_pred.npy")])
        if not pred_files:
            print(f"在目录 {args.pred_dir} 中未找到任何 *_pred.npy 文件。")
            return
        tasks = []
        for pred_filename in pred_files:
            scene_id = pred_filename.replace("_pred.npy", "")
            try:
                area, room = scene_id.split('-', 1)
                full_scene_path = os.path.join(args.data_root, area, room)
            except ValueError:
                parts = scene_id.split('_')
                if len(parts) >= 3:
                    area = parts[0] + '_' + parts[1]
                    room = "_".join(parts[2:])
                    full_scene_path = os.path.join(args.data_root, area, room)
                else:
                    print(f"警告: 无法解析场景ID '{scene_id}'。跳过。")
                    continue
            coords_file = os.path.join(full_scene_path, "coord.npy")
            pred1_filepath = os.path.join(args.pred_dir, pred_filename)
            gt_filepath = os.path.join(full_scene_path, "segment.npy")
            color_filepath = os.path.join(full_scene_path, "color.npy")
            if not os.path.exists(coords_file):
                print(f"警告: 无法找到对应的 coord.npy: '{coords_file}'. 跳过 {scene_id}。")
                continue
            pred2_filepath = None
            if args.compare_pred_dir:
                pred2_filepath = os.path.join(args.compare_pred_dir, pred_filename)
                if not os.path.exists(pred2_filepath):
                    print(f"警告: 无法找到对比预测文件: '{pred2_filepath}'")
                    pred2_filepath = None
            tasks.append({
                'coords': coords_file,
                'pred1_path': pred1_filepath,
                'pred2_path': pred2_filepath,
                'gt_path': gt_filepath if os.path.exists(gt_filepath) else None,
                'color_path': color_filepath if os.path.exists(color_filepath) else None,
                'id': scene_id
            })
        browse_pointclouds_flexible(tasks, args)
    elif args.data_root:
        if not os.path.isdir(args.data_root): 
            raise FileNotFoundError(f"要浏览的数据目录不存在: {args.data_root}")
        tasks = []
        scenes_to_process = []
        for root, _, files in os.walk(args.data_root):
            if "coord.npy" in files:
                scenes_to_process.append(root)
        if not scenes_to_process:
            print(f"在 {args.data_root} 及其子目录中未找到包含 coord.npy 的场景。")
            return
        for scene_path in sorted(scenes_to_process):
            scene_id = f"{Path(scene_path).parent.name}_{Path(scene_path).name}"
            coords_path = os.path.join(scene_path, "coord.npy")
            color_path = os.path.join(scene_path, "color.npy")
            gt_path = os.path.join(scene_path, "segment.npy")
            tasks.append({
                'coords': coords_path,
                'pred1_path': None,
                'pred2_path': None,
                'gt_path': gt_path if os.path.exists(gt_path) else None,
                'color_path': color_path if os.path.exists(color_path) else None,
                'id': scene_id
            })
        browse_pointclouds_flexible(tasks, args)
    else:
        print("错误：无效的参数组合。请选择一种操作模式。")
        parser.print_help()
    print("\n所有点云可视化完成。")

def browse_pointclouds_flexible(tasks, args):
    if not tasks:
        print("没有找到可供可视化的任务。")
        return
    if args.output_dir or args.continuous:
        for i, task in enumerate(tasks):
            print(f"\nProcessing [{i+1}/{len(tasks)}]: {task['id']}...")
            process_and_visualize_flexible(
                task['coords'], args, task['id'],
                pred1_path=task.get('pred1_path'),
                pred2_path=task.get('pred2_path'),
                gt_path=task.get('gt_path'),
                color_path=task.get('color_path')
            )
            if args.continuous and not args.output_dir:
                time.sleep(args.delay)
        return
    index = 0
    while 0 <= index < len(tasks):
        task = tasks[index]
        print(f"\n--- Showing [{index + 1}/{len(tasks)}]: {task['id']} ---")
        process_and_visualize_flexible(
            task['coords'], args, task['id'],
            pred1_path=task.get('pred1_path'),
            pred2_path=task.get('pred2_path'),
            gt_path=task.get('gt_path'),
            color_path=task.get('color_path')
        )
        prompt = "按 Enter 或 'n' 继续下一个, 'p' 查看上一个, 'q' 退出: "
        choice = input(prompt).lower().strip()
        if choice == 'p':
            index = max(0, index - 1)
        elif choice == 'q':
            break
        else:
            index += 1

if __name__ == "__main__":
    main()