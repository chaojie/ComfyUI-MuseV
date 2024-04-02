import warnings
import logging
import os
import pickle
import copy
import time

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from .TransNetmodels import TransNetV2

warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


## 工具函数
def complete_results_batch(
    mp4_ids,
    batch_mp4_scenes_index,
    fps_batch,
    single_frame_pred,
    class_threshold,
    cache_file="/data_share7/v_hyggewang/视频切换模型依赖数据/转场真实标签字典.pkl",
):
    """
    single_frame_pred: [片段数, 100, 2]
    return:[[xs,xs,xs...],[xs,xs..]]每个元素是对应视频的真实值
    """
    cache = pickle.load(open(cache_file, "rb"))  # 读取储存好的MP4真实标签
    pre_index = 0
    result = []

    for mp4_id, index, fps in zip(mp4_ids, batch_mp4_scenes_index, fps_batch):
        raw_transition_index = single_frame_pred[
            pre_index : (pre_index + int(index)), 15:-15, :
        ].reshape(
            -1, 70, 2
        )  # 这里得到15-85，85-155...帧信息具体切割参看dataset中验证集数据生成。

        raw_transition_index = F.softmax(raw_transition_index, dim=-1)  # 获得每一帧对应的类别概率
        zero = torch.zeros_like(raw_transition_index)
        one = torch.ones_like(raw_transition_index)
        raw_transition_index = torch.where(
            raw_transition_index < class_threshold, zero, one
        )[
            :, :, -1
        ]  # 只获取属于1标签的预测结果
        pred_label = raw_transition_index.reshape(-1)  # 得到所有帧的结果

        #         raw_transition_index = F.softmax(raw_transition_index, dim=-1) # 获得每一帧对应的类别概率
        #         pred_label = torch.argmax(raw_transition_index, dim=-1).reshape(-1) # 得到最终类别

        transition_index = (
            torch.where(pred_label == 1)[0] + 15
        ) / fps  # 转场帧位置(前15帧需要加入)

        # 对返回结果做后处理合并相邻帧
        result_transition = []
        for i, transition in enumerate(transition_index):
            if i == 0:
                result_transition.append([transition])
            else:
                if abs(result_transition[-1][-1] - transition) < 0.035:
                    result_transition[-1].append(transition)
                else:
                    result_transition.append([transition])
        result_transition_ = [
            np.mean(item, dtype=np.float16) for item in result_transition
        ]  # 得到最终预测结果

        mp4_GT_label_transition = cache[int(mp4_id)]  # 储存MP4过渡转场真实标签
        result.append({"真实标签": mp4_GT_label_transition, "预测标签": result_transition_})
        pre_index = pre_index + int(index)

    return result


### 工具函数
def pr_call(label_list, thresholds=[0.1, 0.3, 0.5, 0.7]):
    """
    根据时间误差返回各个时间误差情况下的，召回度和准确度
    """
    correct_num_dict = {threshold: 0 for threshold in thresholds}  # 记录各个阈值下准确预测个数
    result = {threshold: None for threshold in thresholds}  # 记录各个阈值下，准确度和召回度
    pre_positive_num = 0  # 所有样本预测正例个数
    GT_positive_num = 0  # 所有样本真实正例个数
    for label_dic in label_list:
        true_labels, pre_labels = label_dic["真实标签"], label_dic["预测标签"]
        pre_positive_num += len(pre_labels)
        GT_positive_num += len(true_labels)

        for threshold in thresholds:
            pre_label_used = set()  # 记录已经匹配的预测标签防止重复匹配
            for true_label in true_labels:
                matched = False  # 真值是否被匹配上了
                for pre_label in pre_labels:
                    if pre_label > true_label + threshold:  # 如果预测值大于了阈值范围，则跳过剩下的预测值
                        break
                    if pre_label in pre_label_used:  # 如果该标签已经被匹配上了则跳过匹配
                        continue
                    if (
                        (true_label - threshold)
                        <= pre_label
                        <= (true_label + threshold)
                    ):
                        correct_num_dict[threshold] += 1
                        matched = True
                    if matched:  # 如果真值已经被匹配上了，则跳过剩下的预测值
                        pre_label_used.add(pre_label)  # 增加已经匹配上的标签
                        break
    for item in correct_num_dict.items():
        result[item[0]] = {
            "precision": item[1] / (pre_positive_num + 1e-8),
            "recall": item[1] / (GT_positive_num + 1e-8),
        }
    return result


class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        logger.info("TransNetV2 模型初始化开始...")
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.lr
        self.model = TransNetV2()

        ## 参数初始化
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        ## 使用原始权重初始化
        if self.args.raw_transnet_weights is not None:
            checkpoint = torch.load(self.args.raw_transnet_weights)
            del checkpoint["cls_layer1.weight"]
            del checkpoint["cls_layer1.bias"]
            del checkpoint["cls_layer2.weight"]
            del checkpoint["cls_layer2.bias"]
            self.model.load_state_dict(checkpoint, strict=False)
            print("载入原始模型权重")

        logger.info("TransNetV2 模型初始化结束")

    def training_step(self, batch, batch_idx):
        frames, one_hot_gt, many_hot_gt = (
            batch["frames"],
            batch["one_hot"],
            batch["many_hot"],
        )
        single_frame_pred, all_frame_pred = self.model(frames)
        return single_frame_pred, all_frame_pred, one_hot_gt, many_hot_gt

    def training_step_end(self, output):
        (
            single_frame_pred,
            all_frame_pred,
            one_hot_gt,
            many_hot_gt,
        ) = output  # single_frame_pred维度为[片段数, 100, 3]，one_hot_gt维度为[片段数, 100]
        loss_one = F.cross_entropy(
            single_frame_pred[:, 15:-15, :].reshape(-1, 2),
            one_hot_gt[:, 15:-15].reshape(-1),
            weight=torch.tensor([0.15, 0.85], device=single_frame_pred.device).type_as(
                single_frame_pred
            ),
        )
        loss_all = F.cross_entropy(
            all_frame_pred[:, 15:-15, :].reshape(-1, 2),
            many_hot_gt[:, 15:-15].reshape(-1),
            weight=torch.tensor([0.15, 0.85], device=all_frame_pred.device).type_as(
                all_frame_pred
            ),
        )
        loss_total = loss_one * 0.9 + loss_all * 0.1

        self.log(
            "train_loss",
            loss_total,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss_total

    def validation_step(self, batch, batch_idx):
        frames, one_hot_gt, many_hot_gt = (
            batch["frames"],
            batch["one_hot"],
            batch["many_hot"],
        )
        single_frame_pred, all_frame_pred = self.model(frames)
        mp4_ids = batch["mp4_ids"]

        batch_mp4_scenes_index = batch["batch_mp4_scenes_index"]
        fps_batch = batch["fps_batch"]
        return (
            single_frame_pred,
            all_frame_pred,
            one_hot_gt,
            many_hot_gt,
            mp4_ids,
            batch_mp4_scenes_index,
            fps_batch,
        )

    def validation_step_end(self, output):
        (
            single_frame_pred,
            all_frame_pred,
            one_hot_gt,
            many_hot_gt,
            mp4_ids,
            _,
            _,
        ) = output

        #         loss_one = self.lossfun(single_frame_pred.reshape(-1,3), one_hot_gt.reshape(-1))
        #         loss_all = self.lossfun(all_frame_pred.reshape(-1,3), many_hot_gt.reshape(-1))
        loss_one = F.cross_entropy(
            single_frame_pred[:, 15:-15, :].reshape(-1, 2),
            one_hot_gt[:, 15:-15].reshape(-1),
            weight=torch.tensor([0.15, 0.85], device=single_frame_pred.device).type_as(
                single_frame_pred
            ),
        )
        loss_all = F.cross_entropy(
            all_frame_pred[:, 15:-15, :].reshape(-1, 2),
            many_hot_gt[:, 15:-15].reshape(-1),
            weight=torch.tensor([0.15, 0.85], device=single_frame_pred.device).type_as(
                single_frame_pred
            ),
        )
        loss_total = loss_one * 0.8 + loss_all * 0.2
        self.log(
            "val_loss",
            loss_total,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
        )

    def validation_epoch_end(self, output):
        start = time.time()
        class_threshold_list = [0.1, 0.3, 0.5, 0.7]
        # 计算每个不同的class_threshold下召准
        for class_threshold in class_threshold_list:
            transition_label_list = []
            for output_each in output:
                (
                    single_frame_pred,
                    all_frame_pred,
                    one_hot_gt,
                    many_hot_gt,
                    mp4_ids,
                    batch_mp4_scenes_index,
                    fps_batch,
                ) = output_each
                transition_label_list = transition_label_list + complete_results_batch(
                    mp4_ids.cpu(),
                    batch_mp4_scenes_index.cpu(),
                    fps_batch.cpu(),
                    single_frame_pred.cpu().float(),
                    class_threshold,
                )
            custom_indicator = pr_call(
                transition_label_list, thresholds=[0.05, 0.1, 0.2, 0.3]
            )
            self.log(
                f"{class_threshold}_0.01s_P",
                custom_indicator[0.05]["precision"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.01s_R",
                custom_indicator[0.05]["recall"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.1s_P",
                custom_indicator[0.1]["precision"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.1s_R",
                custom_indicator[0.1]["recall"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.2s_P",
                custom_indicator[0.2]["precision"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.2s_R",
                custom_indicator[0.2]["recall"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.3s_P",
                custom_indicator[0.3]["precision"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{class_threshold}_0.3s_R",
                custom_indicator[0.3]["recall"],
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
            )
        print("推理耗时:{}".format(time.time() - start))

    ## 优化器配置
    def configure_optimizers(self):
        logger.info("configure_optimizers 初始化开始...")
        # 选择优化器
        if self.args.optim == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # 选择学习率调度方式
        if self.args.lr_scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.0002, verbose=True, epochs=500, steps_per_epoch=7
            )
            logger.info("configure_optimizers 初始化结束...")
            return [optimizer], [scheduler]
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=200, eta_min=5e-7, verbose=True, last_epoch=-1
            )
            logger.info("configure_optimizers 初始化结束...")
            return [optimizer], [scheduler]
        elif self.args.lr_scheduler == "None":
            logger.info("configure_optimizers 初始化结束...")
            return optimizer
