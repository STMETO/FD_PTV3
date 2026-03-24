import torch
import os
import gc
import logging
import sys
import argparse
from datetime import datetime
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def setup_logging(save_path=None):
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if save_path:
        log_dir = os.path.join(save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"test_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件已创建: {log_file}")
    
    return logger


def auto_convert_weights(cfg, logger):
    """
    权重文件自动转换函数
    """
    if not hasattr(cfg, 'weight') or not cfg.weight:
        logger.info("[自动转换] 未在配置中指定权重文件，跳过格式检查。")
        return cfg

    original_path = cfg.weight

    if not os.path.exists(original_path):
        logger.warning(f"[自动转换] 警告：权重文件不存在于: {original_path}，跳过格式检查。")
        return cfg

    logger.info(f"[*] 正在检查权重文件格式: {original_path}")
    try:
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
        
        if 'state_dict' in checkpoint:
            logger.info("[✓] 权重文件格式正确，无需转换。")
            return cfg
        else:
            logger.info("[!] 检测到旧格式权重，将自动进行转换...")
            
            path_without_ext, ext = os.path.splitext(original_path)
            converted_path = f"{path_without_ext}_converted{ext}"
            
            new_checkpoint = {
                'state_dict': checkpoint,
                'epoch': 0
            }
            
            logger.info(f"[*] 正在保存转换后的新文件至: {converted_path}")
            torch.save(new_checkpoint, converted_path)
            
            logger.info(f"[*] 更新配置，后续将使用新文件进行测试。")
            cfg.weight = converted_path

            logger.info("[成功] 权重文件自动转换并更新配置完成！")
            return cfg

    except Exception as e:
        logger.error(f"[自动转换] 错误：处理权重文件时发生异常: {e}")
        return cfg


def apply_custom_config(cfg, logger):
    """
    应用自定义配置
    """
    logger.info("[自定义配置] 开始手动配置...")
    
    # ==================== 配置选项1: 数据加载器配置 ====================
    # 选项A: 内存安全配置（推荐用于内存不足的情况）
    cfg.num_worker = 0      # 工作进程数：0=禁用多进程，避免worker进程内存泄漏
    cfg.batch_size = 1      # 批次大小：1=最小内存占用，适合大场景点云
    
    # 选项B: 性能优化配置（内存充足时使用）
    # cfg.num_worker = 4      # 工作进程数：4=平衡性能与内存
    # cfg.batch_size = 2      # 批次大小：2=适度提升推理速度
    
    # 选项C: 最大性能配置（GPU内存充足时使用）
    # cfg.num_worker = 8      # 工作进程数：8=最大化数据加载速度
    # cfg.batch_size = 4      # 批次大小：4=充分利用GPU并行能力
    
    logger.info(f"  - 设置数据加载器: num_worker={cfg.num_worker}, batch_size={cfg.batch_size}")
    
    # ==================== 配置选项2: 自动混合精度配置 ====================
    # 选项A: 精度优先配置（推荐用于测试评估）
    cfg.enable_amp = False  # False=使用全精度FP32，确保最高计算精度
    
    # 选项B: 速度优先配置（训练或快速推理时使用）
    # cfg.enable_amp = True   # True=使用混合精度FP16，提升推理速度约2倍
    
    # 选项C: 内存优化配置（显存不足时使用）
    # cfg.enable_amp = True   # True=减少显存占用约50%，可能轻微影响精度
    
    if hasattr(cfg, 'enable_amp'):
        logger.info(f"  - 设置自动混合精度: {cfg.enable_amp}")
    
    # ==================== 配置选项3: 模型序列化顺序配置 ====================
    # 选项A: 最高精度配置（推荐用于最终测试）
    # cfg.model.backbone.order = ['z', 'z-trans', 'hilbert', 'hilbert-trans']
    # 说明：完整序列化顺序，提供最全面的空间信息，但计算量最大
    
    # 选项B: 平衡精度速度配置
    cfg.model.backbone.order = ['z', 'hilbert']
    # 说明：减少序列化顺序数量，平衡精度和速度
    
    # 选项C: 最快速度配置（快速测试时使用）
    # cfg.model.backbone.order = ['z']
    # 说明：单一序列化顺序，速度最快但精度可能下降
    
    # 选项D: 内存优化配置（内存不足时使用）
    # cfg.model.backbone.order = ['z']
    # 说明：最小化内存占用的序列化配置
    
    if hasattr(cfg, 'model'):
        if hasattr(cfg.model, 'backbone'):
            if hasattr(cfg.model.backbone, 'order'):
                logger.info(f"  - 设置序列化顺序: {cfg.model.backbone.order}")
    
    # ==================== 配置选项4: 体素大小配置 ====================
    # 选项A: 最高精度配置（原始论文设置）
    # cfg.data.test.test_cfg.voxelize.grid_size = 0.02
    # 说明：0.02=2cm体素，保留最多细节，但内存占用最大
    
    # 选项B: 平衡精度速度配置
    # cfg.data.test.test_cfg.voxelize.grid_size = 0.03
    # 说明：0.03=3cm体素，适度减少点云密度，平衡精度和内存
    
    # 选项C: 内存优化配置（内存不足时使用）
    cfg.data.test.test_cfg.voxelize.grid_size = 0.05
    # 说明：0.05=5cm体素，显著减少内存占用，适合大场景
    
    # 选项D: 最快速度配置（快速测试时使用）
    # cfg.data.test.test_cfg.voxelize.grid_size = 0.08
    # 说明：0.08=8cm体素，最大化处理速度，精度较低
    
    if hasattr(cfg, 'data'):
        if hasattr(cfg.data, 'test'):
            if hasattr(cfg.data.test, 'test_cfg'):
                if hasattr(cfg.data.test.test_cfg, 'voxelize'):
                    logger.info(f"  - 设置体素大小: {cfg.data.test.test_cfg.voxelize.grid_size}")
    
    # ==================== 配置选项5: 测试样本限制配置 ====================
    # 选项A: 完整测试配置（默认，测试所有样本）
    # 不设置任何限制，测试完整数据集
    
    # 选项B: 快速验证配置（调试时使用）
    # if hasattr(cfg.data, 'test'):
    #     cfg.data.test.length = 10  # 只测试前10个样本
    #     logger.info(f"  - 限制测试样本数量: 10")
    
    # 选项C: 单样本测试配置（问题排查时使用）
    # if hasattr(cfg.data, 'test'):
    #     cfg.data.test.length = 1   # 只测试第1个样本
    #     logger.info(f"  - 限制测试样本数量: 1")
    
    # ==================== 配置选项6: 数据增强配置 ====================
    # 选项A: 标准测试配置（禁用数据增强）
    # 测试时通常不需要数据增强，确保结果一致性
    
    # 选项B: TTA测试时增强配置（提升精度）
    # if hasattr(cfg.data, 'test'):
    #     # 启用测试时数据增强，可能提升精度但显著增加计算时间
    #     cfg.data.test.transform = [...复杂的增强配置...]
    #     logger.info("  - 启用测试时数据增强(TTA)")
    
    # ==================== 配置选项7: 内存监控配置 ====================
    # 选项A: 详细内存监控（调试内存问题时使用）
    # cfg.monitor_frequency = 10  # 每10个batch监控一次内存
    
    # 选项B: 最小内存监控（正常运行时使用）
    # cfg.monitor_frequency = 100  # 每100个batch监控一次内存
    
    # 选项C: 禁用内存监控（最大性能）
    # cfg.monitor_frequency = 0   # 禁用内存监控
    
    logger.info("[自定义配置] 配置完成")
    return cfg


def memory_monitor(logger):
    """内存监控函数"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[内存监控] GPU内存: {allocated:.2f}G / {reserved:.2f}G")
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"[内存监控] 系统内存: {memory_info.rss / 1024**3:.2f}G")
    except ImportError:
        logger.warning("[内存监控] 无法导入psutil，跳过系统内存监控")


def cleanup_memory(logger):
    """内存清理函数"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[内存清理] 已执行内存清理")


def create_safe_tester(cfg, logger):
    """创建安全测试器"""
    from pointcept.engines.test import SemSegTester
    
    class SafeSemSegTester(SemSegTester):
        def __init__(self, cfg):
            if not hasattr(cfg, 'weight') or not cfg.weight:
                logger.error("权重文件路径未设置！")
                raise ValueError("权重文件路径未设置，请检查配置")
            
            if not os.path.exists(cfg.weight):
                logger.error(f"权重文件不存在: {cfg.weight}")
                raise FileNotFoundError(f"权重文件不存在: {cfg.weight}")
            
            super().__init__(cfg)
            self.logger = logger
        
        def build_model(self):
            model = super().build_model()
            if not os.path.isfile(self.cfg.weight):
                self.logger.error(f"权重文件不存在: {self.cfg.weight}")
                raise FileNotFoundError(f"权重文件不存在: {self.cfg.weight}")
            return model
    
    return SafeSemSegTester(cfg)


def main_worker(cfg):
    # 设置日志
    save_path = getattr(cfg, 'save_path', './test_result')
    logger = setup_logging(save_path)
    
    logger.info("=" * 50)
    logger.info("开始测试流程")
    logger.info("=" * 50)
    
    # 检查权重文件
    if not hasattr(cfg, 'weight') or not cfg.weight:
        logger.error("错误：权重文件路径未设置！")
        raise ValueError("权重文件路径未设置")
    
    logger.info(f"权重文件路径: {cfg.weight}")
    
    if not os.path.exists(cfg.weight):
        logger.error(f"错误：权重文件不存在: {cfg.weight}")
        raise FileNotFoundError(f"权重文件不存在: {cfg.weight}")
    
    cfg = default_setup(cfg)
    
    # 应用权重转换
    cfg = auto_convert_weights(cfg, logger)
    
    # 应用手动配置
    cfg = apply_custom_config(cfg, logger)
    
    # 清理内存
    cleanup_memory(logger)
    memory_monitor(logger)
    
    # 创建测试器并测试
    try:
        tester = create_safe_tester(cfg, logger)
        logger.info("开始模型测试...")
        tester.test()
        logger.info("测试完成！")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        cleanup_memory(logger)
        raise
    
    cleanup_memory(logger)
    logger.info("测试流程结束")


def custom_argument_parser():
    """自定义参数解析器，支持直接传递权重文件路径"""
    parser = argparse.ArgumentParser(description="FDPTV Testing")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:1234", type=str)
    
    # 直接添加权重文件参数
    parser.add_argument("--weight", default="", help="path to weight file")
    parser.add_argument("--save-path", default="./test_result", help="path to save results")
    parser.add_argument(
        "--options",
        nargs="+",
        help="additional options in key-value pairs"
    )
    return parser


def main():
    # 使用自定义参数解析器
    args = custom_argument_parser().parse_args()
    
    print(f"配置参数:")
    print(f"  Config: {args.config_file}")
    print(f"  Weight: {args.weight}")
    print(f"  Save path: {args.save_path}")
    print(f"  GPU: {args.num_gpus}")
    
    # 检查必要参数
    if not args.config_file:
        print("错误：必须指定配置文件！")
        sys.exit(1)
    
    if not args.weight:
        print("错误：必须指定权重文件！")
        sys.exit(1)
    
    # 解析配置
    cfg = default_config_parser(args.config_file, args.options)
    
    # 直接设置权重和保存路径
    cfg.weight = args.weight
    cfg.save_path = args.save_path
    
    # 启动测试
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
