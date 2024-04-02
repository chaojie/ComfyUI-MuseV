# -*- encoding: utf-8 -*-
# here put the import lib

import os
import sys
import argparse
import glob
import pickle
import json
import cv2
import parser
import random
import pdb
import traceback
import hashlib

from moviepy.editor import VideoFileClip
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.ops import nms

try:
    from ..wenlan.bbox_extractor.bbox_extractor import BboxExtractor
    from ..wenlan.img_feat_extractor import generate_folder_csv
    from ..wenlan.utils import getLanMask
    from ..wenlan.utils.config import cfg_from_yaml_file, cfg
    from ..wenlan.models.vl_model import *
except:
    pass


class ImgModel(nn.Module):
    def __init__(self, model_cfg):
        super(ImgModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable["imgencoder"] = ImgLearnableEncoder(model_cfg)

    def forward(self, imgFea, maskImages, image_boxs):
        imgFea = self.learnable["imgencoder"](
            imgFea, maskImages, image_boxs
        )  # <bsz, img_dim>
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        return imgFea


class ImgFeatureExtractor:
    def __init__(self, cfg_file, model_weights, gpu_id=0):
        self.gpu_id = gpu_id
        self.cfg_file = cfg_file
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.img_model = ImgModel(model_cfg=self.cfg.MODEL)

        self.img_model = self.img_model.cuda(self.gpu_id)
        model_component = torch.load(
            model_weights, map_location=torch.device("cuda:{}".format(self.gpu_id))
        )
        img_model_component = {}
        for key in model_component["learnable"].keys():
            if "imgencoder." in key:
                img_model_component[key] = model_component["learnable"][key]
        self.img_model.learnable.load_state_dict(img_model_component)
        self.img_model.eval()
        self.visual_transform = self.visual_transforms_box(self.cfg.MODEL.IMG_SIZE)

    def visual_transforms_box(self, new_size=456):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((new_size, new_size)), normalize]
        )

    def extract(self, img_path, bboxes):
        if type(img_path) == str:
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.fromarray(img_path)
        if image is None:
            return None
        else:
            width, height = image.size
            new_size = self.cfg.MODEL.IMG_SIZE
            img_box_s = []
            for box_i in bboxes[: self.cfg.MODEL.MAX_IMG_LEN - 1]:  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = (
                    box_i[0] * (new_size / width),
                    box_i[1] * (new_size / height),
                    box_i[2] * (new_size / width),
                    box_i[3] * (new_size / height),
                )
                img_box_s.append(
                    torch.from_numpy(np.array([x1, y1, x2, y2]).astype(np.float32))
                )
            img_box_s.append(
                torch.from_numpy(
                    np.array([0, 0, new_size, new_size]).astype(np.float32)
                )
            )

            image_boxs = torch.stack(img_box_s, 0)  # <36, 4>
            image = self.visual_transform(image)
            img_len = torch.full((1,), self.cfg.MODEL.MAX_IMG_LEN, dtype=torch.long)

            with torch.no_grad():
                imgs = image.unsqueeze(0)  # <batchsize, 3, image_size, image_size>
                img_lens = img_len.unsqueeze(0).view(-1)
                image_boxs = image_boxs.unsqueeze(0)  # <BSZ, 36, 4>

                # get image mask
                imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
                imgMask = imgMask.cuda(self.gpu_id)

                imgs = imgs.cuda(self.gpu_id)
                image_boxs = image_boxs.cuda(self.gpu_id)  # <BSZ, 36, 4>
                img_fea = self.img_model(imgs, imgMask, image_boxs)
                img_fea = img_fea.cpu().numpy()
            return img_fea


def main(video_path, video_map, vf_extractor, bbx_extr):
    video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # video_hash_code = (os.popen('md5sum {}'.format(video_path))).readlines()[0].split('  ')[0]
    with open(video_path, "rb") as fd:
        data = fd.read()
    video_hash_code = hashlib.md5(data).hexdigest()
    save_path = os.path.join(
        save_path, "{}_{}.json".format(video_name, video_hash_code[:8])
    )

    assert video_hash_code == video_map["video_file_hash_code"]

    fps = 1
    max_frame_num = 5
    select_frame_idx = []
    select_frame_clip = []
    for i in range(len(video_map["clips"])):
        clip = video_map["clips"][i]
        if clip["cliptype"] == "transition":
            continue
        select_frame_num = int(min(np.ceil(clip["duration"] * fps), max_frame_num))
        clip_total_frame_num = clip["frame_end"] - clip["frame_start"]
        frame_duration = clip_total_frame_num // (select_frame_num + 1)
        for j in range(select_frame_num):
            select_frame_idx.append(clip["frame_start"] + (j + 1) * frame_duration)
            select_frame_clip.append(i)

    print(len(select_frame_idx), len(set(select_frame_idx)))

    # Capture video
    video = VideoFileClip(video_path)
    video = video.crop(*video_map["content_box"])
    fps = video.fps
    duration = video.duration
    total_frames = int(duration * fps)
    width, height = video.size
    print("fps, frame_count, width, height:", fps, total_frames, width, height)

    cnt_frame, step = 0, 0
    for frame in video.iter_frames(fps=video_map["sample_fps"]):
        if step == len(select_frame_idx):
            break
        if cnt_frame == select_frame_idx[step]:
            bboxes = bbx_extr.extract_bboxes(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            bboxes = bboxes.tolist()
            fea = vf_extractor.extract(frame, bboxes)
            fea = fea.squeeze(axis=0).tolist()
            if "feat" in video_map["clips"][select_frame_clip[step]]:
                video_map["clips"][select_frame_clip[step]]["feat"].append(fea)
            else:
                video_map["clips"][select_frame_clip[step]]["feat"] = [fea]

            step += 1
        cnt_frame += 1

    for clip in video_map["clips"]:
        clip["multi_factor"] = {"semantics": None}
        if "feat" in clip:
            clip["multi_factor"]["semantics"] = np.mean(
                np.array(clip["feat"]), axis=0
            ).tolist()
    return video_map


class WenLanVisualPredictor(object):
    def __init__(
        self,
        brivl_checkpoint,
        cfg_file="cfg/test_xyb.yml",
        bbox_extractor_cfg="bbox_extractor/configs/bua-caffe/extract-bua-caffe-r101.yaml",
    ) -> None:
        # brivl_checkpoint = '/innovation_cfs/mmatch/infguo/weights/BriVL-1.0-5500w.pth',
        abs_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(abs_path, cfg_file)
        bbox_extractor_cfg = os.path.join(abs_path, bbox_extractor_cfg)
        self.vf_extractor = ImgFeatureExtractor(cfg_file, brivl_checkpoint)
        self.bbx_extr = BboxExtractor(bbox_extractor_cfg)

    def __call__(
        self,
        video_path,
        video_map,
    ):
        video_map = main(video_path, video_map, self.vf_extractor, self.bbx_extr)
        return video_map
