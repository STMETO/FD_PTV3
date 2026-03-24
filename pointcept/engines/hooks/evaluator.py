"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


# @HOOKS.register_module()
# class SemSegEvaluator(HookBase):
#     def __init__(self, write_cls_iou=False):
#         self.write_cls_iou = write_cls_iou

#     def before_train(self):
#         if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
#             wandb.define_metric("val/*", step_metric="Epoch")

#     def after_epoch(self):
#         if self.trainer.cfg.evaluate:
#             self.eval()

#     def eval(self):
#         self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
#         self.trainer.model.eval()
#         for i, input_dict in enumerate(self.trainer.val_loader):
#             for key in input_dict.keys():
#                 if isinstance(input_dict[key], torch.Tensor):
#                     input_dict[key] = input_dict[key].cuda(non_blocking=True)
#             with torch.no_grad():
#                 output_dict = self.trainer.model(input_dict)
#             output = output_dict["seg_logits"]
#             loss = output_dict["loss"]
#             pred = output.max(1)[1]
#             segment = input_dict["segment"]
#             if "inverse" in input_dict.keys():
#                 assert "origin_segment" in input_dict.keys()
#                 pred = pred[input_dict["inverse"]]
#                 segment = input_dict["origin_segment"]
#             intersection, union, target = intersection_and_union_gpu(
#                 pred,
#                 segment,
#                 self.trainer.cfg.data.num_classes,
#                 self.trainer.cfg.data.ignore_index,
#             )
#             if comm.get_world_size() > 1:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
#                     target
#                 )
#             intersection, union, target = (
#                 intersection.cpu().numpy(),
#                 union.cpu().numpy(),
#                 target.cpu().numpy(),
#             )
#             # Here there is no need to sync since sync happened in dist.all_reduce
#             self.trainer.storage.put_scalar("val_intersection", intersection)
#             self.trainer.storage.put_scalar("val_union", union)
#             self.trainer.storage.put_scalar("val_target", target)
#             self.trainer.storage.put_scalar("val_loss", loss.item())
#             info = "Test: [{iter}/{max_iter}] ".format(
#                 iter=i + 1, max_iter=len(self.trainer.val_loader)
#             )
#             if "origin_coord" in input_dict.keys():
#                 info = "Interp. " + info
#             self.trainer.logger.info(
#                 info
#                 + "Loss {loss:.4f} ".format(
#                     iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
#                 )
#             )
#         loss_avg = self.trainer.storage.history("val_loss").avg
#         intersection = self.trainer.storage.history("val_intersection").total
#         union = self.trainer.storage.history("val_union").total
#         target = self.trainer.storage.history("val_target").total
#         iou_class = intersection / (union + 1e-10)
#         acc_class = intersection / (target + 1e-10)
#         m_iou = np.mean(iou_class)
#         m_acc = np.mean(acc_class)
#         all_acc = sum(intersection) / (sum(target) + 1e-10)
#         self.trainer.logger.info(
#             "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
#                 m_iou, m_acc, all_acc
#             )
#         )
#         for i in range(self.trainer.cfg.data.num_classes):
#             self.trainer.logger.info(
#                 "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
#                     idx=i,
#                     name=self.trainer.cfg.data.names[i],
#                     iou=iou_class[i],
#                     accuracy=acc_class[i],
#                 )
#             )
#         current_epoch = self.trainer.epoch + 1
#         if self.trainer.writer is not None:
#             self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
#             self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
#             self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
#             self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
#             if self.trainer.cfg.enable_wandb:
#                 wandb.log(
#                     {
#                         "Epoch": current_epoch,
#                         "val/loss": loss_avg,
#                         "val/mIoU": m_iou,
#                         "val/mAcc": m_acc,
#                         "val/allAcc": all_acc,
#                     },
#                     step=wandb.run.step,
#                 )
#             if self.write_cls_iou:
#                 for i in range(self.trainer.cfg.data.num_classes):
#                     self.trainer.writer.add_scalar(
#                         f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
#                         iou_class[i],
#                         current_epoch,
#                     )
#                 if self.trainer.cfg.enable_wandb:
#                     for i in range(self.trainer.cfg.data.num_classes):
#                         wandb.log(
#                             {
#                                 "Epoch": current_epoch,
#                                 f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
#                                     i
#                                 ],
#                             },
#                             step=wandb.run.step,
#                         )
#         self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
#         self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
#         self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

#     def after_train(self):
#         self.trainer.logger.info(
#             "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
#         )


# @HOOKS.register_module()
# class SemSegEvaluator(HookBase):
#     """
#     语义分割评估器钩子 (Hook)。

#     这个钩子在每个 epoch 结束后执行评估，计算 mIoU、mAcc 等指标，
#     并将结果记录到 TensorBoard 和 Weights & Biases (wandb)。
#     它适用于本地客户端的周期性验证，并能确保 wandb 曲线在联邦学习的多轮中连续。
#     """
#     def __init__(self, write_cls_iou=False):
#         """
#         初始化函数。
#         Args:
#             write_cls_iou (bool): 如果为 True,则单独记录每个类别的 IoU。
#         """
#         self.write_cls_iou = write_cls_iou

#     def before_train(self):
#         """
#         在训练开始前执行，用于为 wandb 定义指标的X轴。
#         """
#         # 这部分定义了 wandb 图表的X轴，对于本地和全局 Run 都有好处，可以保留
#         if self.trainer.writer is not None and self.trainer.cfg.get("enable_wandb", False):
#             # 为训练过程中的指标定义X轴
#             wandb.define_metric("train_step")
#             wandb.define_metric("train_batch/*", step_metric="train_step")
#             wandb.define_metric("lr", step_metric="train_step")
            
#             # 为周期性/验证指标定义X轴
#             wandb.define_metric("global_step")
#             wandb.define_metric("val_local/*", step_metric="global_step")


#     def after_epoch(self):
#         """
#         在每个 epoch 结束后执行，用于触发评估流程。
#         """
#         if self.trainer.cfg.get("evaluate", False):
#             self.eval()

#     def after_train(self):
#         """
#         在整个训练过程结束后执行，用于记录最佳指标。
#         """
#         if hasattr(self.trainer, "best_metric_value") and self.trainer.best_metric_value is not None:
#             self.trainer.logger.info(
#                 f"最佳 mIoU: {self.trainer.best_metric_value:.4f}"
#             )

#     def eval(self):
#         """
#         核心评估逻辑。
#         """
#         self.trainer.logger.info(">>>>>>>>>>>>>>>> 开始本地验证 >>>>>>>>>>>>>>>>")
#         self.trainer.model.eval()

#         # 计算一个在联邦学习多轮中能够连续递增的X轴步骤
#         if hasattr(self.trainer.cfg, "current_round"):
#             global_step = self.trainer.cfg.current_round * self.trainer.max_epoch + self.trainer.epoch + 1
#         else:
#             global_step = self.trainer.epoch + 1

#         for i, input_dict in enumerate(self.trainer.val_loader):
#             for key, value in input_dict.items():
#                 if isinstance(value, torch.Tensor):
#                     input_dict[key] = value.cuda(non_blocking=True)
            
#             with torch.no_grad():
#                 output_dict = self.trainer.model(input_dict)

#             loss = output_dict["loss"]
#             pred = output_dict["seg_logits"].max(1)[1]
#             segment = input_dict["segment"]

#             if "origin_coord" in input_dict:
#                 idx, _ = pointops.knn_query(
#                     1,
#                     input_dict["coord"].float(),
#                     input_dict["offset"].int(),
#                     input_dict["origin_coord"].float(),
#                     input_dict["origin_offset"].int(),
#                 )
#                 pred = pred[idx.flatten().long()]
#                 segment = input_dict["origin_segment"]

#             intersection, union, target = intersection_and_union_gpu(
#                 pred,
#                 segment,
#                 self.trainer.cfg.data.num_classes,
#                 self.trainer.cfg.data.ignore_index,
#             )

#             if comm.get_world_size() > 1:
#                 dist.all_reduce(intersection)
#                 dist.all_reduce(union)
#                 dist.all_reduce(target)

#             # 使用 pointcept 的 storage 工具来累积结果
#             self.trainer.storage.put_scalar("val_intersection", intersection.cpu().numpy())
#             self.trainer.storage.put_scalar("val_union", union.cpu().numpy())
#             self.trainer.storage.put_scalar("val_target", target.cpu().numpy())
#             self.trainer.storage.put_scalar("val_loss", loss.item())

#             # 构建并打印当前验证批次的进度信息
#             info = f"验证进度: [{i + 1}/{len(self.trainer.val_loader)}]"
#             if "origin_coord" in input_dict:
#                 # 如果存在 origin_coord，说明是插值过程
#                 info = "插值. " + info
#             info += f" 损失 {loss.item():.4f}"
#             self.trainer.logger.info(info)

#         # 从 storage 中获取整个验证集的汇总结果
#         loss_avg = self.trainer.storage.history("val_loss").avg
#         intersection = self.trainer.storage.history("val_intersection").total
#         union = self.trainer.storage.history("val_union").total
#         target = self.trainer.storage.history("val_target").total

#         iou_class = intersection / (union + 1e-10)
#         acc_class = intersection / (target + 1e-10)
#         m_iou = np.mean(iou_class[target > 0]) # 修正：只对有目标的类计算均值
#         m_acc = np.mean(acc_class[target > 0]) # 修正：只对有目标的类计算均值
#         all_acc = sum(intersection) / (sum(target) + 1e-10)

#         self.trainer.logger.info(
#             f"本地验证结果: mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}."
#         )

#         # 1. 写入 TensorBoard (保持不变)
#         if self.trainer.writer is not None:
#             self.trainer.writer.add_scalar("val/loss", loss_avg, global_step)
#             self.trainer.writer.add_scalar("val/mIoU", m_iou, global_step)
#             self.trainer.writer.add_scalar("val/mAcc", m_acc, global_step)
#             self.trainer.writer.add_scalar("val/allAcc", all_acc, global_step)

#         # 2. 写入 wandb (修正后的逻辑)
#         if self.trainer.cfg.get("enable_wandb", False):
#             # 使用 "val_local/" 前缀来明确这是本地验证指标
#             wandb_log_data = {
#                 "val_local/loss": loss_avg,
#                 "val_local/mIoU": m_iou,
#                 "val_local/mAcc": m_acc,
#                 "val_local/allAcc": all_acc,
#             }
            
#             if self.write_cls_iou:
#                 for i in range(self.trainer.cfg.data.num_classes):
#                     if target[i] > 0: # 只记录有意义的类别
#                         key_name = f"val_local/cls_{i}-{self.trainer.cfg.data.names[i]} IoU"
#                         wandb_log_data[key_name] = iou_class[i]
            
#             # 使用 step 参数指定X轴，确保曲线在多轮联邦学习中连续
#             wandb.log(wandb_log_data, step=global_step)

#         self.trainer.logger.info("<<<<<<<<<<<<<<<<< 结束本地验证 <<<<<<<<<<<<<<<<<")
        
#         self.trainer.comm_info["current_metric_value"] = m_iou
#         self.trainer.comm_info["current_metric_name"] = "mIoU"

@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """
    语义分割评估器钩子 (Hook)。

    这个钩子在每个 epoch 结束后执行评估，计算 mIoU、mAcc 等指标，
    并将结果记录到 TensorBoard 和 Weights & Biases (wandb)。
    支持普通训练和联邦学习两种模式，提供完整的类别级别监控。
    """
    def __init__(self, write_cls_iou=True, write_cls_acc=True):
        """
        初始化函数。
        Args:
            write_cls_iou (bool): 如果为 True,则单独记录每个类别的 IoU。
            write_cls_acc (bool): 如果为 True,则单独记录每个类别的准确率。
        """
        self.write_cls_iou = write_cls_iou
        self.write_cls_acc = write_cls_acc

    def before_train(self):
        """
        在训练开始前执行，用于为 wandb 定义指标的X轴。
        """
        # 这部分定义了 wandb 图表的X轴，对于本地和全局 Run 都有好处，可以保留
        if self.trainer.writer is not None and self.trainer.cfg.get("enable_wandb", False):
            # 为训练过程中的指标定义X轴
            wandb.define_metric("train_step")
            wandb.define_metric("train_batch/*", step_metric="train_step")
            wandb.define_metric("lr", step_metric="train_step")
            
            # 为周期性/验证指标定义X轴
            wandb.define_metric("global_step")
            wandb.define_metric("val_local/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")  # 新增：普通训练验证指标
            wandb.define_metric("val_Class_IoU/*", step_metric="global_step")  # 新增：类别IoU
            wandb.define_metric("val_Class_Acc/*", step_metric="global_step")  # 新增：类别准确率
            wandb.define_metric("val_local_Class_IoU/*", step_metric="global_step")  # 新增：联邦学习类别IoU
            wandb.define_metric("val_local_Class_Acc/*", step_metric="global_step")  # 新增：联邦学习类别准确率

    def after_epoch(self):
        """
        在每个 epoch 结束后执行，用于触发评估流程。
        """
        if self.trainer.cfg.get("evaluate", False):
            self.eval()

    def after_train(self):
        """
        在整个训练过程结束后执行，用于记录最佳指标。
        """
        if hasattr(self.trainer, "best_metric_value") and self.trainer.best_metric_value is not None:
            self.trainer.logger.info(
                f"最佳 mIoU: {self.trainer.best_metric_value:.4f}"
            )

    def eval(self):
        """
        核心评估逻辑 - 增强版本，支持完整的类别级别监控。
        """
        self.trainer.logger.info(">>>>>>>>>>>>>>>> 开始验证 >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        # 判断是否为联邦学习模式
        is_federated = hasattr(self.trainer.cfg, "current_round")
        
        # 计算一个在联邦学习多轮中能够连续递增的X轴步骤
        if is_federated:
            global_step = self.trainer.cfg.current_round * self.trainer.max_epoch + self.trainer.epoch + 1
            prefix = "val_local"  # 联邦学习使用 val_local 前缀
        else:
            global_step = self.trainer.epoch + 1
            prefix = "val"  # 普通训练使用 val 前缀

        # 初始化统计量用于累积整个验证集的结果
        num_classes = self.trainer.cfg.data.num_classes
        class_names = getattr(self.trainer.cfg.data, "names", [f"cls_{i}" for i in range(num_classes)])
        
        intersection_total = np.zeros(num_classes)
        union_total = np.zeros(num_classes)
        target_total = np.zeros(num_classes)
        loss_total = 0.0
        n_batches = 0

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)
            
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            pred = output_dict["seg_logits"].max(1)[1]
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
                pred,
                segment,
                num_classes,
                self.trainer.cfg.data.ignore_index,
            )

            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            # 累积统计量
            intersection_total += intersection.cpu().numpy()
            union_total += union.cpu().numpy()
            target_total += target.cpu().numpy()
            loss_total += loss.item()
            n_batches += 1

            # 构建并打印当前验证批次的进度信息
            info = f"验证进度: [{i + 1}/{len(self.trainer.val_loader)}]"
            if "origin_coord" in input_dict:
                # 如果存在 origin_coord，说明是插值过程
                info = "插值. " + info
            info += f" 损失 {loss.item():.4f}"
            self.trainer.logger.info(info)

        # 计算最终指标
        loss_avg = loss_total / max(n_batches, 1)
        iou_class = intersection_total / (union_total + 1e-10)
        acc_class = intersection_total / (target_total + 1e-10)
        m_iou = np.mean(iou_class[target_total > 0])
        m_acc = np.mean(acc_class[target_total > 0])
        all_acc = np.sum(intersection_total) / (np.sum(target_total) + 1e-10)

        self.trainer.logger.info(
            f"验证结果: mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}."
        )

        # 1. 写入 TensorBoard
        if self.trainer.writer is not None:
            # 整体指标
            self.trainer.writer.add_scalar(f"{prefix}/loss", loss_avg, global_step)
            self.trainer.writer.add_scalar(f"{prefix}/mIoU", m_iou, global_step)
            self.trainer.writer.add_scalar(f"{prefix}/mAcc", m_acc, global_step)
            self.trainer.writer.add_scalar(f"{prefix}/allAcc", all_acc, global_step)

            # 类别级别指标
            if self.write_cls_iou or self.write_cls_acc:
                for i in range(num_classes):
                    if target_total[i] > 0:
                        cls_name = class_names[i] if i < len(class_names) else f"cls_{i}"
                        if self.write_cls_iou:
                            self.trainer.writer.add_scalar(f"{prefix}_Class_IoU/{cls_name}", iou_class[i], global_step)
                        if self.write_cls_acc:
                            self.trainer.writer.add_scalar(f"{prefix}_Class_Acc/{cls_name}", acc_class[i], global_step)

        # 2. 写入 wandb
        if self.trainer.cfg.get("enable_wandb", False):
            # 基础指标
            wandb_log_data = {
                f"{prefix}/loss": loss_avg,
                f"{prefix}/mIoU": m_iou,
                f"{prefix}/mAcc": m_acc,
                f"{prefix}/allAcc": all_acc,
            }
            
            # 类别级别指标
            if self.write_cls_iou or self.write_cls_acc:
                for i in range(num_classes):
                    if target_total[i] > 0:
                        cls_name = class_names[i] if i < len(class_names) else f"cls_{i}"
                        if self.write_cls_iou:
                            wandb_log_data[f"{prefix}_Class_IoU/{cls_name}"] = iou_class[i]
                        if self.write_cls_acc:
                            wandb_log_data[f"{prefix}_Class_Acc/{cls_name}"] = acc_class[i]
            
            # 使用 step 参数指定X轴
            wandb.log(wandb_log_data, step=global_step)

        # 详细日志输出 - 显示每个类别的性能
        if self.write_cls_iou or self.write_cls_acc:
            self.trainer.logger.info("各类别详细性能:")
            for i in range(num_classes):
                if target_total[i] > 0:
                    cls_name = class_names[i] if i < len(class_names) else f"cls_{i}"
                    info_str = f"  {cls_name}:"
                    if self.write_cls_iou:
                        info_str += f" IoU={iou_class[i]:.4f}"
                    if self.write_cls_acc:
                        info_str += f" Acc={acc_class[i]:.4f}"
                    self.trainer.logger.info(info_str)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< 结束验证 <<<<<<<<<<<<<<<<<")
        
        self.trainer.comm_info["current_metric_value"] = m_iou
        self.trainer.comm_info["current_metric_name"] = "mIoU"


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mAP": all_ap,
                        "val/AP50": all_ap_50,
                        "val/AP25": all_ap_25,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
