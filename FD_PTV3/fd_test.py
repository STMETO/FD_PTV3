"""
FD_PTV3 联邦学习测试入口
========================
用于对联邦学习训练的最终模型进行测试评估。
保持与原 FDPTV_test.py 一致的测试逻辑。

用法:
    python -m FD_PTV3.fd_test --config-file configs/fedavg_s3dis.py --weight path/to/final_model.pth
"""

import os
import sys
import gc
import logging
import argparse
import torch
from datetime import datetime

from pointcept.engines.defaults import (
    default_config_parser,
)
from pointcept.engines.launch import launch


def setup_logging(save_path=None):
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
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
    """权重文件自动转换"""
    if not hasattr(cfg, 'weight') or not cfg.weight:
        logger.info("[自动转换] 未指定权重文件，跳过格式检查。")
        return cfg

    original_path = cfg.weight
    if not os.path.exists(original_path):
        logger.warning(f"[自动转换] 权重文件不存在: {original_path}")
        return cfg

    logger.info(f"[*] 正在检查权重文件格式: {original_path}")
    try:
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
        if 'state_dict' in checkpoint:
            logger.info("[✓] 权重文件格式正确")
            return cfg

        logger.info("[!] 检测到旧格式权重，自动转换...")
        path_without_ext, ext = os.path.splitext(original_path)
        converted_path = f"{path_without_ext}_converted{ext}"

        new_checkpoint = {'state_dict': checkpoint, 'epoch': 0}
        torch.save(new_checkpoint, converted_path)
        logger.info(f"[*] 已保存转换文件: {converted_path}")
        cfg.weight = converted_path
        logger.info("[成功] 权重文件转换完成")
        return cfg
    except Exception as e:
        logger.error(f"[自动转换] 异常: {e}")
        return cfg


def cleanup_memory(logger):
    """内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[内存清理] 已执行")


def create_safe_tester(cfg, logger):
    """创建安全测试器"""
    from pointcept.engines.test import SemSegTester

    class SafeSemSegTester(SemSegTester):
        def __init__(self, cfg):
            if not hasattr(cfg, 'weight') or not cfg.weight:
                raise ValueError("权重文件路径未设置")
            if not os.path.exists(cfg.weight):
                raise FileNotFoundError(f"权重文件不存在: {cfg.weight}")
            super().__init__(cfg)
            self.logger = logger

        def build_model(self):
            model = super().build_model()
            if not os.path.isfile(self.cfg.weight):
                raise FileNotFoundError(f"权重文件不存在: {self.cfg.weight}")
            return model

    return SafeSemSegTester(cfg)


def main_worker(cfg):
    save_path = getattr(cfg, 'save_path', './fd_test_result')
    logger = setup_logging(save_path)

    logger.info("=" * 50)
    logger.info("FD_PTV3 开始测试流程")
    logger.info("=" * 50)

    if not hasattr(cfg, 'weight') or not cfg.weight:
        logger.error("错误：权重文件路径未设置！")
        raise ValueError("权重文件路径未设置")

    logger.info(f"权重文件路径: {cfg.weight}")
    if not os.path.exists(cfg.weight):
        logger.error(f"错误：权重文件不存在: {cfg.weight}")
        raise FileNotFoundError(f"权重文件不存在: {cfg.weight}")

    from pointcept.engines.defaults import default_setup
    cfg = default_setup(cfg)
    cfg = auto_convert_weights(cfg, logger)

    cleanup_memory(logger)

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
    """自定义参数解析器"""
    parser = argparse.ArgumentParser(description="FD_PTV3 Testing")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:1234", type=str)
    parser.add_argument("--weight", default="", help="path to weight file")
    parser.add_argument("--save-path", default="./fd_test_result", help="path to save results")
    parser.add_argument("--options", nargs="+", help="additional options in key-value pairs")
    return parser


def main():
    args = custom_argument_parser().parse_args()

    print(f"配置参数:")
    print(f"  Config: {args.config_file}")
    print(f"  Weight: {args.weight}")
    print(f"  Save path: {args.save_path}")

    if not args.config_file:
        print("错误：必须指定配置文件！")
        sys.exit(1)
    if not args.weight:
        print("错误：必须指定权重文件！")
        sys.exit(1)

    cfg = default_config_parser(args.config_file, args.options)
    cfg.weight = args.weight
    cfg.save_path = args.save_path

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
