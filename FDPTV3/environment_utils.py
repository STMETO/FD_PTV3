# environment_utils.py
import os
import logging
import shutil
from torch.utils.tensorboard import SummaryWriter

def setup_environment(cfg):
    """
    初始化训练环境，包括设置全局日志记录器和 TensorBoard。

    Args:
        cfg (object/dict): 主配置对象。

    Returns:
        tuple: 包含 (glogger, writer, save_path) 的元组。
    """
    save_path = cfg.get("save_path", "./") if isinstance(cfg, dict) else getattr(cfg, "save_path", "./")
    global_log_file = os.path.join(save_path, "federated_training.log")
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S", 
        handlers=[
            logging.FileHandler(global_log_file, mode="a"), 
            logging.StreamHandler()
        ]
    )
    glogger = logging.getLogger("global_logger")
    writer_path = os.path.join(save_path, "fed_model_tensorboard") 
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)
    return glogger, writer, save_path

def cleanup_previous_artifacts(save_path, glogger):
    """
    清理上一次单机运行时可能残留的日志和模型文件。

    Args:
        save_path (str): 保存路径。
        glogger (Logger): 全局日志记录器。
    """
    model_dir = os.path.join(save_path, "model")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        glogger.info("已清理旧的单机模型目录")
    
    log_file = os.path.join(save_path, "train_user_-1.log")
    if os.path.isfile(log_file):
        os.remove(log_file)
        glogger.info("已清理旧的单机日志文件")

def cleanup_client_checkpoints(save_path, num_users, glogger):
    """
    在一轮聚合后,删除所有客户端的本地模型检查点(model_last.pth)。

    Args:
        save_path (str): 主保存路径。
        num_users (int): 客户端总数。
        glogger (Logger): 全局日志记录器。
    """
    glogger.info("清理本轮所有客户端的本地检查点...")
    for i in range(num_users):
        client_checkpoint = os.path.join(save_path, f"user_{i}", "model", "model_last.pth")
        if os.path.exists(client_checkpoint):
            try: 
                os.remove(client_checkpoint)
            except Exception as e: 
                glogger.warning(f"[警告] 删除用户 {i + 1} 的检查点 {client_checkpoint} 失败: {e}")