import os
import numpy as np
import open3d as o3d
import argparse
import time
from pathlib import Path

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
    """
    将实例分割标签映射回语义分割标签
    """
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
            mask = (instance_labels == instance_id).flatten()  # 确保是一维布尔数组
            semantic_labels[mask] = class_id
        return semantic_labels
    else:
        print("警告: 未找到实例到类别的映射文件，使用自动映射")
        semantic_labels = instance_labels % NUM_CLASSES
        return semantic_labels

# --- 为实例分割生成独特颜色的函数 ---
def get_instance_colors(instance_labels):
    """为实例分割生成独特的颜色"""
    if instance_labels.size == 0:
        return np.empty((0, 3))
    
    # 确保instance_labels是一维数组
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
        
        # 修复：确保布尔索引是一维的
        mask = (instance_labels == instance_id)
        if mask.ndim > 1:
            mask = mask.flatten()
        colors[mask] = rgb
    
    return colors

# --- 修改后的 get_label_colors 函数 ---
def get_label_colors(labels, label_type='semantic', instance2class_file=None):
    """根据标签ID生成颜色。"""
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

    # 确保labels是一维数组
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
    """打印颜色和对应标签的图例"""
    if labels.size == 0:
        return
    
    # 确保labels是一维数组
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

# --- 灵活的多窗口可视化函数 ---
def visualize_flexible_pointclouds(coords, window_configs, scene_id, args):
    """
    根据窗口配置灵活显示多个窗口
    window_configs: 列表，每个元素是 (window_type, colors, window_name)
    """
    try:
        # 检查数据形状是否一致
        print(f"坐标形状: {coords.shape}")
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            print(f"窗口 {i+1} ({window_type}) 颜色形状: {colors.shape}")
        
        # 确保所有颜色数组形状正确
        def ensure_correct_shape(colors, expected_points, name):
            if colors.ndim == 3:
                print(f"调整 {name} 形状: {colors.shape} -> ({colors.shape[0]}, 3)")
                colors = colors.reshape(-1, 3)
            
            if colors.shape[0] != expected_points:
                print(f"警告: {name} 数据点数量 ({colors.shape[0]}) 与坐标数据点数量 ({expected_points}) 不匹配")
                if colors.shape[0] < expected_points:
                    repeat_times = expected_points // colors.shape[0] + 1
                    colors = np.tile(colors, (repeat_times, 1))[:expected_points]
                else:
                    colors = colors[:expected_points]
            
            return colors
        
        # 调整所有颜色数组形状
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            window_configs[i] = (window_type, 
                               ensure_correct_shape(colors, coords.shape[0], f"{window_name}颜色"), 
                               window_name)
        
        # 创建点云对象和可视化器
        pcds = []
        visualizers = []
        
        for window_type, colors, window_name in window_configs:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pcd)
            
            vis = o3d.visualization.Visualizer()
            visualizers.append(vis)
        
        # 设置窗口位置和大小
        window_width = args.win_width
        window_height = args.win_height
        
        # 根据窗口数量智能调整大小
        num_windows = len(window_configs)
        if num_windows > 4:
            window_width = min(window_width, 400)
            window_height = min(window_height, 350)
        elif num_windows > 2:
            window_width = min(window_width, 450)
            window_height = min(window_height, 400)
        
        # 智能布局算法
        screen_width = 1920  # 假设标准屏幕宽度
        screen_height = 1080  # 假设标准屏幕高度
        
        # 计算网格布局
        if num_windows <= 2:
            cols = num_windows
            rows = 1
        elif num_windows <= 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = (num_windows + cols - 1) // cols
        
        # 计算窗口间距和起始位置
        horizontal_gap = 20
        vertical_gap = 40
        start_x = (screen_width - (cols * window_width + (cols - 1) * horizontal_gap)) // 2
        start_y = 50
        
        # 创建窗口并设置位置
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
            
            # 设置渲染选项
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([0.0, 0.0, 0.0])
        
        # 显示窗口信息
        print(f"显示 {num_windows} 个窗口对比: {scene_id}")
        print("窗口布局:")
        for i, (window_type, colors, window_name) in enumerate(window_configs):
            print(f"  窗口 {i+1}: {window_name}")
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
            for vis in visualizers:
                vis.destroy_window()
                
    except Exception as e:
        import traceback
        print(f"在多窗口可视化过程中出错: {str(e)}")
        traceback.print_exc()
        raise

# --- 单窗口可视化函数 ---
def visualize_single_pointcloud(coords, colors, title="", output_path=None, delay=0, win_width=1280, win_height=720):
    """可视化单个点云，支持保存图片或实时显示。"""
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
            opt.background_color = np.asarray([0.0, 0.0, 0.0])
            vis.run()
        finally:
            vis.destroy_window()

        if delay > 0:
            time.sleep(delay)

# --- 指标计算函数 ---
def calculate_and_print_metrics(pred_labels, gt_labels, pred_name="预测"):
    """根据预测标签和真值标签，计算并打印详细的分割指标。"""
    # 确保标签是一维数组
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

# --- 核心处理函数（重构为支持灵活窗口配置）---
def process_and_visualize_flexible(coords_path, args, scene_id, 
                                  pred1_path=None, pred2_path=None, 
                                  gt_path=None, color_path=None,
                                  instance_path=None):
    """
    灵活的可视化处理函数，支持任意窗口组合
    """
    try:
        coords = np.load(coords_path)
        
        # 准备窗口配置
        window_configs = []
        
        # 解析 solid_color 参数（用于 blank 窗口）
        try:
            solid_color = [float(x) for x in args.solid_color.split(',')]
            if max(solid_color) > 1.0:
                solid_color = [c / 255.0 for c in solid_color]
            if len(solid_color) != 3:
                raise ValueError
        except:
            print(f"警告：无效的 solid_color 参数 '{args.solid_color}'，使用默认灰色 [0.5,0.5,0.5]")
            solid_color = [0.5, 0.5, 0.5]
        
        # 解析窗口类型
        window_types = args.windows.split(',') if args.windows else []
        
        # 加载所需数据并创建窗口配置
        for window_type in window_types:
            window_type = window_type.strip().lower()
            
            if window_type == 'color':
                # 原始颜色数据
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
                # 真值语义数据
                if gt_path and os.path.exists(gt_path):
                    print(f"加载真值文件: {gt_path}")
                    gt_labels = np.load(gt_path)
                    gt_colors = get_label_colors(gt_labels, label_type='semantic')
                    window_configs.append(('gt', gt_colors, "真值语义"))
                    
                    # 打印真值图例
                    print_color_legend(gt_labels, "semantic")
                else:
                    print(f"警告: 真值文件不存在: {gt_path}，跳过此窗口")
            
            elif window_type == 'instance':
                # 实例分割数据
                if instance_path is None:
                    instance_path = os.path.join(os.path.dirname(coords_path), "instance.npy")
                
                if os.path.exists(instance_path):
                    print(f"加载实例文件: {instance_path}")
                    instance_labels = np.load(instance_path)
                    
                    # 查找实例到类别的映射文件
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
                    
                    # 打印实例图例
                    print_color_legend(instance_labels, "instance", instance2class_file)
                else:
                    print(f"警告: 实例文件不存在: {instance_path}，跳过此窗口")
            
            elif window_type == 'pred1':
                # 第一个预测数据
                if pred1_path and os.path.exists(pred1_path):
                    print(f"加载预测1文件: {pred1_path}")
                    pred1_labels = np.load(pred1_path)
                    if pred1_labels.ndim == 2 and pred1_labels.shape[1] == 1:
                        pred1_labels = pred1_labels.flatten()
                    pred1_colors = get_label_colors(pred1_labels, label_type='semantic')
                    window_configs.append(('pred1', pred1_colors, args.pred1_name))
                    
                    # 计算指标（如果有真值）
                    if gt_path and os.path.exists(gt_path):
                        gt_labels = np.load(gt_path)
                        calculate_and_print_metrics(pred1_labels, gt_labels, args.pred1_name)
                    
                    # 打印预测1图例
                    print(f"\n--- {args.pred1_name} 图例 ---")
                    print_color_legend(pred1_labels, "semantic")
                else:
                    print(f"警告: 预测1文件不存在: {pred1_path}，跳过此窗口")
            
            elif window_type == 'pred2':
                # 第二个预测数据
                if pred2_path and os.path.exists(pred2_path):
                    print(f"加载预测2文件: {pred2_path}")
                    pred2_labels = np.load(pred2_path)
                    if pred2_labels.ndim == 2 and pred2_labels.shape[1] == 1:
                        pred2_labels = pred2_labels.flatten()
                    pred2_colors = get_label_colors(pred2_labels, label_type='semantic')
                    window_configs.append(('pred2', pred2_colors, args.pred2_name))
                    
                    # 计算指标（如果有真值）
                    if gt_path and os.path.exists(gt_path):
                        gt_labels = np.load(gt_path)
                        calculate_and_print_metrics(pred2_labels, gt_labels, args.pred2_name)
                    
                    # 打印预测2图例
                    print(f"\n--- {args.pred2_name} 图例 ---")
                    print_color_legend(pred2_labels, "semantic")
                else:
                    print(f"警告: 预测2文件不存在: {pred2_path}，跳过此窗口")
            
            elif window_type == 'blank':
                # 纯色窗口
                colors = np.full((coords.shape[0], 3), solid_color)
                window_configs.append(('blank', colors, "纯色"))
            
            else:
                print(f"警告: 未知的窗口类型 '{window_type}'，跳过")
        
        # 如果没有指定窗口或所有窗口都无效，使用默认单窗口显示
        if not window_configs:
            print("没有有效的窗口配置，使用默认单窗口显示")
            colors = np.full((coords.shape[0], 3), 0.5)
            title = f"默认视图 - {scene_id}"
            output_path = os.path.join(args.output_dir, f"{scene_id}_default.png") if args.output_dir else None
            visualize_single_pointcloud(
                coords, colors, title=title, output_path=output_path, delay=args.delay,
                win_width=args.win_width, win_height=args.win_height
            )
            return True
        
        # 显示多窗口
        print("开始多窗口可视化...")
        visualize_flexible_pointclouds(coords, window_configs, scene_id, args)
        return True

    except Exception as e:
        import traceback
        print(f"Error processing scene '{scene_id}': {str(e)}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="灵活的点云可视化工具 (支持任意窗口组合，包括纯色窗口)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 核心参数
    parser.add_argument("--coords", type=str, default=None, help="模式1: 单个点云坐标文件路径 (coord.npy)。")
    parser.add_argument("--pred", type=str, default=None, help="模式1: 与 --coords 配合使用，提供单个预测标签文件路径。")
    parser.add_argument("--pred_dir", type=str, default=None, help="模式2: 包含预测标签 (*_pred.npy) 的目录。")
    parser.add_argument("--data_root", type=str, default=None, help="模式2 & 3: 原始数据的根目录。")
    
    # 窗口配置参数
    parser.add_argument("--windows", type=str, default="", 
                    help="指定要显示的窗口类型和顺序，用逗号分隔。\n"
                            "可用类型: color(原始颜色), gt(真值语义), instance(实例分割), pred1(预测1), pred2(预测2), blank(纯色)\n"
                            "示例: --windows color,gt,pred1,pred2\n"
                            "      --windows color,instance\n"
                            "      --windows gt,instance,pred1\n"
                            "      --windows blank               # 只显示纯色窗口\n"
                            "      --windows color,blank,gt      # 混合显示")
    
    # 预测配置参数
    parser.add_argument("--compare_pred_dir", type=str, default=None, help="用于第二个预测的目录")
    parser.add_argument("--pred1_name", type=str, default="预测1", help="第一个预测的名称")
    parser.add_argument("--pred2_name", type=str, default="预测2", help="第二个预测的名称")
    
    # 纯色窗口参数
    parser.add_argument("--solid_color", type=str, default="0.5,0.5,0.5",
                    help="纯色窗口的颜色，格式为 R,G,B，取值范围 0~1 或 0~255。例如 '0.7,0.2,0.9'")
    
    # 其他参数
    parser.add_argument("--mode", type=str, default="segment", choices=["color", "segment", "instance"], 
                       help="用于模式1和模式3，指定要可视化的原始数据类型。")
    parser.add_argument("--output_dir", type=str, default=None, help="可选：保存可视化图像的目录。")
    parser.add_argument("--continuous", action="store_true", help="如果设置，将自动连续显示所有点云。")
    parser.add_argument("--delay", type=float, default=0.5, help="在连续显示模式下，每个点云显示后的延迟时间（秒）。")
    parser.add_argument("--win_width", type=int, default=500, help="设置交互式显示窗口的宽度。")
    parser.add_argument("--win_height", type=int, default=500, help="设置交互式显示窗口的高度。")
    
    # 向后兼容的参数
    parser.add_argument("--three_windows", action="store_true", help="向后兼容：等同于 --windows color,gt,pred1")
    parser.add_argument("--four_windows", action="store_true", help="向后兼容：等同于 --windows color,gt,pred1,pred2")
    
    args = parser.parse_args()
    
    # 处理向后兼容性
    if args.three_windows and not args.windows:
        args.windows = "color,gt,pred1"
        print("使用三窗口模式: color,gt,pred1")
    
    if args.four_windows and not args.windows:
        args.windows = "color,gt,pred1,pred2"
        print("使用四窗口模式: color,gt,pred1,pred2")
    
    # 窗口大小调整
    if args.windows:
        num_windows = len(args.windows.split(','))
        if num_windows > 4:
            args.win_width = min(args.win_width, 400)
            args.win_height = min(args.win_height, 350)
            print(f"多窗口模式：自动调整窗口大小为 {args.win_width}x{args.win_height}")
        elif num_windows > 2:
            args.win_width = min(args.win_width, 450)
            args.win_height = min(args.win_height, 400)
            print(f"多窗口模式：自动调整窗口大小为 {args.win_width}x{args.win_height}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"可视化图像将保存到: {args.output_dir}")

    # 模式一: 单个文件可视化
    if args.coords:
        if not os.path.exists(args.coords): 
            raise FileNotFoundError(f"坐标文件不存在: {args.coords}")
        
        scene_id = Path(args.coords).parent.name
        data_dir = Path(args.coords).parent
        
        # 构建文件路径
        gt_path = data_dir / "segment.npy"
        color_path = data_dir / "color.npy"
        instance_path = data_dir / "instance.npy"  
        
        pred1_path = None
        pred2_path = None
        
        if args.pred:
            pred1_path = args.pred
        elif args.windows and any(w in args.windows for w in ['pred1', 'pred']):
            # 如果没有明确指定pred但窗口配置需要pred1，使用默认预测文件
            pred1_path = data_dir / "pred.npy"
            if not pred1_path.exists():
                print(f"警告: 默认预测文件不存在: {pred1_path}")
        
        # 修复：正确处理 compare_pred_dir 参数
        if args.compare_pred_dir:
            # 检查是否是一个文件
            if os.path.isfile(args.compare_pred_dir):
                # 如果提供的是文件路径，直接使用
                pred2_path = args.compare_pred_dir
                print(f"使用指定的预测2文件: {pred2_path}")
            else:
                # 如果提供的是目录，构建文件路径
                pred2_filename = f"{scene_id}_pred.npy"
                pred2_path = os.path.join(args.compare_pred_dir, pred2_filename)
                print(f"构建预测2文件路径: {pred2_path}")
            
            if not os.path.exists(pred2_path):
                print(f"警告: 预测2文件不存在: {pred2_path}，跳过此窗口")
        
        # 使用灵活的可视化函数
        process_and_visualize_flexible(
            args.coords, args, scene_id,
            pred1_path=pred1_path,
            pred2_path=pred2_path,
            gt_path=str(gt_path) if gt_path.exists() else None,
            color_path=str(color_path) if color_path.exists() else None,
            instance_path=str(instance_path) if instance_path.exists() else None  
        )

    # 模式二: 批量可视化预测结果
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
            
            # 构建第二个预测路径（如果提供）
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

    # 模式三: 批量浏览原始数据
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
        if not args.coords and not (args.pred_dir and args.data_root) and not args.data_root:
            print("错误：无效的参数组合。请选择一种操作模式。")
            parser.print_help()

    print("\n所有点云可视化完成。")

def browse_pointclouds_flexible(tasks, args):
    """通用浏览控制器，支持前进、后退和退出。"""
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