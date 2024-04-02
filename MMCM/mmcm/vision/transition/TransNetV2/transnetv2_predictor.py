# -*- coding: utf-8 -*-
from typing import Dict, List, Union
import os
import json
import traceback
import argparse
import hashlib

import librosa
import soundfile as sf
import numpy as np
import torch
from moviepy.editor import *

from .TransNetmodels import TransNetV2
from ...video_map.video_meta_info import VideoMetaInfo
from ...video_map.video_map import VideoMap
from ...video_map.video_clip import VideoClipSeq
from ...black_border import det_video_black_border
from ...utils.path_util import get_video_signature
from ...data.video_dataset import MoviepyVideoDataset, SequentialDataset


def predict(
    model,
    video_path,
    threshold=0.3,
    sample_fps=25,
    content_box=None,
    single_frame_ratio=1,
    map_path: str = None,
    ignored_keys: List = None,
) -> VideoMap:
    video_hash_code, video_path = get_video_signature(path=video_path, rename=True)
    basename = os.path.basename(video_path)
    filename, ext = os.path.splitext(video_path)
    with torch.no_grad():
        (
            single_frame_result,
            all_frame_result,
            fps,
            total_frames,
            duration,
            height,
            width,
        ) = model.predict_video(
            video_path,
            cache_path="",
            c_box=content_box,
            width=48,
            height=27,
            input_frames=10000,
            overlap=100,
            sample_fps=sample_fps,
        )
    # pred_label = single_frame_ratio * single_frame_result + (1 - single_frame_ratio) * all_frame_result
    pred_label = np.array([single_frame_result, all_frame_result])
    pred_label = pred_label.max(axis=0)
    transition_index = np.where(pred_label > threshold)[0]  # 转场帧位置
    transition_index = transition_index.astype(np.float)
    # 对返回结果做后处理合并相邻帧
    result_transition = []
    for i, transition in enumerate(transition_index):
        if i == 0:
            result_transition.append([transition])
        else:
            if abs(result_transition[-1][-1] - transition) <= 4:
                result_transition[-1].append(transition)
            else:
                result_transition.append([transition])

    result = [[0]]
    for item in result_transition:
        start_idx = int(item[0])
        end_idx = int(item[-1])
        if len(item) > 3:
            if max(pred_label[start_idx : end_idx + 1]) > 0.3:
                result.append([item[0], item[-1]])
        elif len(item) > 1:
            if max(pred_label[start_idx : end_idx + 1]) > 0.4:
                result.append([item[0], item[-1]])
        else:
            if pred_label[start_idx] > 0.45:
                result.append(item)
    result.append([pred_label.shape[0]])

    video_meta_info_dct = {
        "video_name": filename,
        "video_path": video_path,
        "video_file_hash_code": video_hash_code,
        "fps": fps,
        "frame_num": total_frames,
        "duration": duration,
        "height": height,
        "width": width,
        "content_box": content_box,
        "sample_fps": sample_fps,
    }
    video_meta_info = VideoMetaInfo.from_video_path(video_path)
    video_meta_info.__dict__.update(video_meta_info_dct)

    video_clipseq = []
    slice_id = 0
    for i in range(len(result) - 1):
        if len(result[i]) == 1:
            vidoe_clip = {
                "time_start": round(result[i][0] / sample_fps, 4),  # 开始时间
                "duration": round(
                    result[i + 1][0] / sample_fps - result[i][0] / sample_fps,
                    4,
                ),  # 片段持续时间
                "frame_start": result[i][0],
                "frame_end": result[i + 1][0],
                "clipid": slice_id,  # 片段序号，
                "cliptype": "body",
            }
            video_clipseq.append(vidoe_clip)
            slice_id += 1
        elif len(result[i]) == 2:
            vidoe_clip = {
                "time_start": round(result[i][0] / sample_fps, 4),  # 开始时间
                "duration": round(
                    result[i][1] / sample_fps - result[i][0] / sample_fps,
                    4,
                ),  # 片段持续时间
                "frame_start": result[i][0],
                "frame_end": result[i][1],
                "clipid": slice_id,  # 片段序号，
                "cliptype": "transition",
            }
            video_clipseq.append(vidoe_clip)
            slice_id += 1

            vidoe_clip = {
                "time_start": round(result[i][1] / sample_fps, 4),  # 开始时间
                "duration": round(
                    result[i + 1][0] / sample_fps - result[i][1] / sample_fps,
                    4,
                ),  # 片段持续时间
                "frame_start": result[i][1],
                "frame_end": result[i + 1][0],
                "clipid": slice_id,  # 片段序号，
                "cliptype": "body",
            }
            video_clipseq.append(vidoe_clip)
            slice_id += 1
    video_clipseq = VideoClipSeq.from_data(video_clipseq)
    video_map = VideoMap(meta_info=video_meta_info, clipseq=video_clipseq)
    if map_path is not None:
        with open(map_path, "w") as f:
            json.dump(video_map.to_dct(ignored_keys=ignored_keys), f, indent=4)
    return video_map, single_frame_result, all_frame_result


class TransNetV2Predictor(object):
    def __init__(self, model_path: str, device: str) -> None:
        # 模型初始化和参数载入
        self.model = TransNetV2()
        checkpoint = torch.load(model_path)  # 载入模型参数
        self.model.load_state_dict(
            {k.replace("model.", ""): v for k, v in checkpoint.items()}
        )
        # model.load_state_dict(checkpoint['state_dict'])
        self.model.eval().to(device)
        self.device = device

    def __call__(self, video_path, map_path, content_box) -> Dict:
        return predict(
            self.model, video_path, map_path=map_path, content_box=content_box
        )

    # TODO: is writing
    def predict_video_write(
        self,
        video_dataset: Union[str, SequentialDataset],
        c_box=None,
        width=48,
        height=27,
        input_frames=100,
        overlap=30,
        sample_fps=30,
        threshold=0.3,
        drop_last=False,
    ):
        # check parameters
        assert overlap % 2 == 0
        assert input_frames > overlap

        # prepare video_dataset
        if isinstance(video_dataset, str):
            video_dataset = MoviepyVideoDataset(video_dataset)
        step = input_frames - overlap
        if (
            video_dataset.step != step
            or video_dataset.time_size != input_frames
            or video_dataset.drop_last != drop_last
        ):
            video_dataset.generate_sample_idxs(
                time_size=input_frames, step=step, drop_last=drop_last
            )
        fps = video_dataset.fps
        duration = video_dataset.duration
        total_frames = video_dataset.total_frames
        w, h = video_dataset.size

        if c_box:
            video_dataset.cap.crop(*c_box)

        single_frame_pred_lst, all_frame_pred_lst, index_lst = [], [], []
        for i, batch in enumerate(video_dataset):
            data, data_index = batch.data, batch.index
            data = data.to(self.device)
            # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
            single_frame_pred, all_frame_pred = self.forward(data.unsqueeze(0))  # 前向推理
            # single_frame_pred = F.softmax(single_frame_pred, dim=-1) # 获得每一帧对应的类别概率
            # single_frame_pred = torch.argmax(single_frame_pred, dim=-1).reshape(-1)
            single_frame_pred = torch.sigmoid(single_frame_pred).reshape(-1)
            all_frame_pred = torch.sigmoid(all_frame_pred).reshape(-1)
            # single_frame_pred = (single_frame_pred>threshold)*1
            if total_frames > data_index[-1]:
                if i == 0:
                    single_frame_pred_label = single_frame_pred[: -overlap // 2]
                    all_frame_pred_label = all_frame_pred[: -overlap // 2]
                else:
                    single_frame_pred_label = single_frame_pred[
                        overlap // 2 : -overlap // 2
                    ]
                    all_frame_pred_label = all_frame_pred[overlap // 2 : -overlap // 2]
            else:
                if i == 0:
                    single_frame_pred_label = single_frame_pred
                    all_frame_pred_label = all_frame_pred
                else:
                    single_frame_pred_label = single_frame_pred[overlap // 2 :]
                    all_frame_pred_label = all_frame_pred[overlap // 2 :]
            single_frame_pred_lst.append(single_frame_pred_label)
            all_frame_pred_lst.append(all_frame_pred_label)
            index_lst.extent(data_index)
        single_frame_pred_label = torch.concat(single_frame_pred_lst, dim=0)
        all_frame_pred_label = torch.concat(all_frame_pred_lst, dim=0)
        single_frame_pred_label = single_frame_pred_label.cpu().numpy()
        all_frame_pred_label = all_frame_pred_label.cpu().numpy()

        # 对返回结果做后处理合并相邻帧
        pred_label = np.array([single_frame_pred_label, all_frame_pred_label])
        pred_label = pred_label.max(axis=0)
        transition_index = np.where(pred_label > threshold)[0]  # 转场帧位置
        transition_index = transition_index.astype(np.float)
        result_transition = []
        for i, transition in enumerate(transition_index):
            if i == 0:
                result_transition.append([transition])
            else:
                if abs(result_transition[-1][-1] - transition) <= 4:
                    result_transition[-1].append(transition)
                else:
                    result_transition.append([transition])

        result = [[0]]
        for item in result_transition:
            start_idx = int(item[0])
            end_idx = int(item[-1])
            if len(item) > 3:
                if max(pred_label[start_idx : end_idx + 1]) > 0.3:
                    result.append([item[0], item[-1]])
            elif len(item) > 1:
                if max(pred_label[start_idx : end_idx + 1]) > 0.4:
                    result.append([item[0], item[-1]])
            else:
                if pred_label[start_idx] > 0.45:
                    result.append(item)
        result.append([pred_label.shape[0]])

        return (
            single_frame_pred_label,
            all_frame_pred_label,
            fps,
            total_frames,
            duration,
            h,
            w,
        )
