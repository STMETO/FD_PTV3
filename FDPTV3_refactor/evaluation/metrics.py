"""联邦模型验证评估逻辑。"""

import numpy as np
import torch

import pointcept.utils.comm as comm
import pointops
from pointcept.utils.misc import intersection_and_union_gpu

from ..utils.config import _get_cfg


def eval_fed_model(model, val_loader, writer, logger, round_idx, cfg):
    """验证联邦聚合模型，写入 TensorBoard 和 WandB。

    round_idx 在这个函数里约定为对外展示轮次，即 1-based。
    """
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    num_classes = _get_cfg(cfg, "data.num_classes")
    ignore_index = _get_cfg(cfg, "data.ignore_index")
    class_names = _get_cfg(cfg, "data.names")

    intersection_total = np.zeros(num_classes)
    union_total = np.zeros(num_classes)
    target_total = np.zeros(num_classes)
    loss_total = 0.0
    n_batches = 0

    if len(val_loader) == 0:
        logger.warning(f"验证集 DataLoader 为空，跳过第 {round_idx} 轮评估。")
        if writer is not None:
            writer.add_text("Warning", f"Validation skipped for round {round_idx}: DataLoader is empty.", round_idx)
        return 0.0, 0.0, 0.0, 0.0

    logger.info(f"正在对第 {round_idx} 轮联邦聚合模型进行评估...")

    with torch.no_grad():
        for input_dict in val_loader:
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            output_dict = model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "origin_coord" in input_dict:
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred, segment, num_classes, ignore_index
            )

            if comm.get_world_size() > 1 and torch.distributed.is_initialized():
                import torch.distributed as dist

                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            intersection_total += intersection.cpu().numpy()
            union_total += union.cpu().numpy()
            target_total += target.cpu().numpy()
            loss_total += loss.item()
            n_batches += 1

    loss_avg = loss_total / max(n_batches, 1)

    iou_class = intersection_total / (union_total + 1e-10)
    acc_class = intersection_total / (target_total + 1e-10)

    iou_class[np.isnan(iou_class)] = 0.0
    acc_class[np.isnan(acc_class)] = 0.0
    iou_class[np.isinf(iou_class)] = 0.0
    acc_class[np.isinf(acc_class)] = 0.0

    valid_mask = target_total > 0
    m_iou = np.mean(iou_class[valid_mask]) if np.any(valid_mask) else 0.0
    m_acc = np.mean(acc_class[valid_mask]) if np.any(valid_mask) else 0.0
    all_acc = np.sum(intersection_total) / (np.sum(target_total) + 1e-10)

    if writer is not None:
        writer.add_scalar("Fed_model_val/loss", loss_avg, round_idx)
        writer.add_scalar("Fed_model_val/mIoU", m_iou, round_idx)
        writer.add_scalar("Fed_model_val/mAcc", m_acc, round_idx)
        writer.add_scalar("Fed_model_val/allAcc", all_acc, round_idx)

        if class_names is not None and len(class_names) == num_classes:
            for index in range(num_classes):
                if target_total[index] > 0:
                    writer.add_scalar(f"Fed_model_val_Class_IoU/{class_names[index]}", iou_class[index], round_idx)
                    writer.add_scalar(f"Fed_model_val_Class_Acc/{class_names[index]}", acc_class[index], round_idx)

    if _get_cfg(cfg, "enable_wandb", False):
        import wandb

        wandb_metrics = {
            "Fed_model_val/loss": loss_avg,
            "Fed_model_val/mIoU": m_iou,
            "Fed_model_val/mAcc": m_acc,
            "Fed_model_val/allAcc": all_acc,
        }
        if class_names is not None and len(class_names) == num_classes:
            for index in range(num_classes):
                if target_total[index] > 0:
                    wandb_metrics[f"Fed_model_val_Class_IoU/{class_names[index]}"] = iou_class[index]
                    wandb_metrics[f"Fed_model_val_Class_Acc/{class_names[index]}"] = acc_class[index]
        wandb.log(wandb_metrics, step=round_idx)

    logger.info(f"第 {round_idx} 轮联邦聚合模型验证结果:")
    logger.info(f"  mIoU={m_iou:.4f}, mAcc={m_acc:.4f}, allAcc={all_acc:.4f}, loss={loss_avg:.4f}")

    if class_names is not None and len(class_names) == num_classes:
        for index in range(num_classes):
            if target_total[index] > 0:
                logger.info(f"  Class_{index}-{class_names[index]}: iou={iou_class[index]:.4f}, acc={acc_class[index]:.4f}")

    return m_iou, m_acc, all_acc, loss_avg