
# val_writer.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging # 确保 logging 模块导入，因为 glogger 是外部传入的

# 导入 pointcept 框架的相关模块。
# 假设这些是 pointcept 环境中的标准依赖，且已正确安装。
import pointcept.utils.comm as comm
import pointops
from pointcept.utils.misc import intersection_and_union_gpu

# 辅助函数，保持与主训练脚本一致的 cfg 访问方式
def _get_cfg(cfg, key, default=None):
    try:
        return getattr(cfg, key)
    except Exception:
        return cfg.get(key, default)

def eval_fed_model(model, val_loader, writer, logger, round_idx, cfg): # round 改为 round_idx 更具描述性
    """
    验证联邦聚合模型性能，并写入 TensorBoard
    Args:
        model: torch.nn.Module, 聚合后的全局模型
        val_loader: DataLoader, 验证集
        writer: SummaryWriter, TensorBoard 写入器
        logger: logging.Logger, 日志记录器
        round_idx: int, 当前全局轮数
        cfg: 配置对象
    Returns:
        m_iou, m_acc, all_acc, loss_avg
    """
    # 确保模型在正确的设备上，虽然在主脚本中已处理，这里增加健壮性
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # 从配置中获取评估所需的参数
    num_classes = _get_cfg(cfg, "data")["num_classes"]
    ignore_index = _get_cfg(cfg, "data")["ignore_index"]
    class_names = _get_cfg(cfg, "data")["names"] # 假设存在

    intersection_total = np.zeros(num_classes)
    union_total = np.zeros(num_classes)
    target_total = np.zeros(num_classes)
    loss_total = 0.0
    n_batches = 0

    # 优化点1: 处理 val_loader 为空的情况
    if len(val_loader) == 0:
        logger.warning(f"验证集DataLoader为空,跳过第 {round_idx} 轮评估。")
        if writer is not None:
            writer.add_text("Warning", f"Validation skipped for round {round_idx}: DataLoader is empty.", round_idx)
        return 0.0, 0.0, 0.0, 0.0 # 返回默认值

    logger.info(f"正在对第 {round_idx} 轮联邦聚合模型进行评估...")

    with torch.no_grad():
        for i, input_dict in enumerate(val_loader):
            # 将 tensor 移动到 GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            
            # 前向传播
            output_dict = model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            # 处理可能的 origin_coord
            if "origin_coord" in input_dict:
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int()
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            # 计算交并比
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                num_classes, # 使用从 cfg 获取的 num_classes
                ignore_index # 使用从 cfg 获取的 ignore_index
            )

            # 多 GPU 汇总
            if comm.get_world_size() > 1:
                # 确保在分布式模式下，所有设备上的结果都被正确聚合
                if torch.distributed.is_initialized(): # 增加检查以避免在非分布式环境中报错
                    import torch.distributed as dist
                    dist.all_reduce(intersection)
                    dist.all_reduce(union)
                    dist.all_reduce(target)
                else:
                    logger.warning("Tried to all_reduce in non-initialized distributed environment. Skipping.")

            intersection_total += intersection.cpu().numpy()
            union_total += union.cpu().numpy()
            target_total += target.cpu().numpy()
            loss_total += loss.item()
            n_batches += 1

    # 汇总指标
    loss_avg = loss_total / max(n_batches, 1) # 确保 n_batches 不为0
    
    iou_class = intersection_total / (union_total + 1e-10)
    acc_class = intersection_total / (target_total + 1e-10)

    # 过滤掉 NaN 或 Inf 值，这些可能因为 union_total 或 target_total 为 0 导致
    iou_class[np.isnan(iou_class)] = 0.0
    acc_class[np.isnan(acc_class)] = 0.0
    iou_class[np.isinf(iou_class)] = 0.0
    acc_class[np.isinf(acc_class)] = 0.0

    m_iou = np.mean(iou_class[target_total > 0]) # 只计算有样本的类别的平均值
    m_acc = np.mean(acc_class[target_total > 0]) # 只计算有样本的类别的平均值
    all_acc = np.sum(intersection_total) / (np.sum(target_total) + 1e-10)

    # 写入 TensorBoard
    if writer is not None:
        writer.add_scalar("Fed_model_val/loss", loss_avg, round_idx)
        writer.add_scalar("Fed_model_val/mIoU", m_iou, round_idx)
        writer.add_scalar("Fed_model_val/mAcc", m_acc, round_idx)
        writer.add_scalar("Fed_model_val/allAcc", all_acc, round_idx)

        # 优化点6: 为每个类别添加 IoU 到 TensorBoard
        if class_names is not None and len(class_names) == num_classes:
            for i in range(num_classes):
                # 只有当该类别有目标（即非忽略类别且有数据）时才记录其 IoU
                if target_total[i] > 0:
                    writer.add_scalar(f"Fed_model_val_Class_IoU/{class_names[i]}", iou_class[i], round_idx)
                    writer.add_scalar(f"Fed_model_val_Class_Acc/{class_names[i]}", acc_class[i], round_idx)
                else:
                    logger.debug(f"类别 {class_names[i]} 在本轮验证中没有目标，未记录其 IoU/Acc。")

    # 2. 写入 wandb
    if _get_cfg(cfg, "enable_wandb", False):
        import wandb
        
        # 创建一个字典来收集所有要记录的指标
        wandb_metrics = {
            "Fed_model_val/loss": loss_avg,
            "Fed_model_val/mIoU": m_iou,
            "Fed_model_val/mAcc": m_acc,
            "Fed_model_val/allAcc": all_acc,
        }
        
        # 添加每个类别的 IoU 和 Acc
        if class_names is not None and len(class_names) == num_classes:
            for i in range(num_classes):
                if target_total[i] > 0:
                    wandb_metrics[f"Fed_model_val_Class_IoU/{class_names[i]}"] = iou_class[i]
                    wandb_metrics[f"Fed_model_val_Class_Acc/{class_names[i]}"] = acc_class[i]
        
        # 一次性将所有指标记录到 wandb，并指定 step
        wandb.log(wandb_metrics, step=round_idx)

    # 日志输出
    logger.info(f"第 {round_idx} 轮联邦聚合模型验证结果:")
    logger.info(f"  mIoU={m_iou:.4f}, mAcc={m_acc:.4f}, allAcc={all_acc:.4f}, loss={loss_avg:.4f}")
    
    # 优化点5: 调整详细类别日志的输出级别
    if class_names is not None and len(class_names) == num_classes:
        for i in range(num_classes):
            # 只有当该类别有目标时才打印详细信息
            if target_total[i] > 0:
                logger.info(f"  Class_{i}-{class_names[i]}: iou={iou_class[i]:.4f}, acc={acc_class[i]:.4f}")
            else:
                logger.debug(f"  Class_{i}-{class_names[i]}: 无目标样本，跳过 IoU/Acc 打印。")
    else:
        # 如果没有类别名称，或者长度不匹配，只打印 IoU/Acc
        for i in range(num_classes):
             if target_total[i] > 0:
                logger.info(f"  Class_{i}: iou={iou_class[i]:.4f}, acc={acc_class[i]:.4f}")
             else:
                logger.debug(f"  Class_{i}: 无目标样本，跳过 IoU/Acc 打印。")


    return m_iou, m_acc, all_acc, loss_avg