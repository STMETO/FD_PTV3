"""最终模型测试器。"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime

import torch

from pointcept.engines.defaults import default_setup


def setup_logging(save_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")

    return logger


def auto_convert_weights(cfg, logger):
    if not hasattr(cfg, "weight") or not cfg.weight:
        logger.info("[自动转换] 未指定权重文件，跳过格式检查。")
        return cfg

    original_path = cfg.weight
    if not os.path.exists(original_path):
        logger.warning(f"[自动转换] 权重文件不存在: {original_path}")
        return cfg

    logger.info(f"[*] 正在检查权重文件格式: {original_path}")
    try:
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            logger.info("[✓] 权重文件格式正确")
            return cfg

        logger.info("[!] 检测到旧格式权重，自动转换...")
        path_without_ext, ext = os.path.splitext(original_path)
        converted_path = f"{path_without_ext}_converted{ext}"
        torch.save({"state_dict": checkpoint, "epoch": 0}, converted_path)
        cfg.weight = converted_path
        logger.info(f"[*] 已保存转换文件: {converted_path}")
        return cfg
    except Exception as exc:
        logger.error(f"[自动转换] 异常: {exc}")
        return cfg


def cleanup_memory(logger):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[内存清理] 已执行")


class FinalModelTester:
    """最终模型测试封装。"""

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(self):
        from pointcept.engines.test import SemSegTester

        class SafeSemSegTester(SemSegTester):
            def __init__(self, cfg, logger):
                if not hasattr(cfg, "weight") or not cfg.weight:
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

        tester = SafeSemSegTester(self.cfg, self.logger)
        self.logger.info("开始模型测试...")
        tester.test()
        self.logger.info("测试完成！")


def main_worker(cfg):
    save_path = getattr(cfg, "save_path", "./fd_test_result")
    logger = setup_logging(save_path)
    logger.info("=" * 50)
    logger.info("FDPTV3_refactor 开始测试流程")
    logger.info("=" * 50)

    if not hasattr(cfg, "weight") or not cfg.weight:
        raise ValueError("权重文件路径未设置")
    if not os.path.exists(cfg.weight):
        raise FileNotFoundError(f"权重文件不存在: {cfg.weight}")

    cfg = default_setup(cfg)
    cfg = auto_convert_weights(cfg, logger)
    cleanup_memory(logger)

    try:
        FinalModelTester(cfg, logger).run()
    except Exception:
        cleanup_memory(logger)
        raise

    cleanup_memory(logger)
    logger.info("测试流程结束")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="FDPTV3_refactor Testing")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:1234", type=str)
    parser.add_argument("--weight", default="", help="path to weight file")
    parser.add_argument("--save-path", default="./fd_test_result", help="path to save results")
    parser.add_argument("--options", nargs="+", help="additional options in key-value pairs")
    return parser
