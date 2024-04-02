# -*- encoding: utf-8 -*-
# here put the import lib

import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import pickle
import json
import cv2
import parser
import pandas as pd
import random
import pdb
from torchvision.ops import nms
import traceback
from moviepy.editor import VideoFileClip
import hashlib

from bbox_extractor.bbox_extractor import BboxExtractor
from img_feat_extractor import generate_folder_csv
from utils import getLanMask
from utils.config import cfg_from_yaml_file, cfg
from models.vl_model import *


class ImgModel(nn.Module):
    def __init__(self, model_cfg):
        super(ImgModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg)

    def forward(self, imgFea, maskImages, image_boxs):
        imgFea = self.learnable['imgencoder'](imgFea, maskImages, image_boxs)  # <bsz, img_dim>
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        return imgFea


class ImgFeatureExtractor:
    def __init__(self, cfg_file, model_weights, gpu_id=0):
        self.gpu_id = gpu_id
        self.cfg_file = cfg_file
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.img_model = ImgModel(model_cfg=self.cfg.MODEL)

        self.img_model = self.img_model.cuda(self.gpu_id)
        model_component = torch.load(model_weights, map_location=torch.device('cuda:{}'.format(self.gpu_id)))
        img_model_component = {}
        for key in model_component["learnable"].keys():
            if "imgencoder." in key:
                img_model_component[key] = model_component["learnable"][key]
        self.img_model.learnable.load_state_dict(img_model_component)
        self.img_model.eval()
        self.visual_transform = self.visual_transforms_box(self.cfg.MODEL.IMG_SIZE)

    def visual_transforms_box(self, new_size=456):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_size, new_size)),
            normalize])

    def extract(self, img_path, bboxes):
        if type(img_path)==str:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(img_path)
        if image is None:
            return None
        else:
            width, height = image.size
            new_size = self.cfg.MODEL.IMG_SIZE
            img_box_s = []
            for box_i in bboxes[:self.cfg.MODEL.MAX_IMG_LEN - 1]:  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box_i[0] * (new_size / width), box_i[1] * (new_size / height), box_i[2] * (
                            new_size / width), box_i[3] * (new_size / height)
                img_box_s.append(torch.from_numpy(np.array([x1, y1, x2, y2]).astype(np.float32)))
            img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32)))

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

def main(video_path, save_path, map_path, vf_extractor, bbx_extr):

    video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # video_hash_code = (os.popen('md5sum {}'.format(video_path))).readlines()[0].split('  ')[0]
    with open(video_path, 'rb') as fd:
        data = fd.read()
    video_hash_code = hashlib.md5(data).hexdigest()
    save_path = os.path.join(save_path, '{}_{}.json'.format(video_name, video_hash_code[:8]))
    if os.path.exists(save_path) and not args.overwrite:
        print('exists ' + save_path)
        pass
    else:
        map_path = os.path.join(map_path, '{}_{}.json'.format(video_name, video_hash_code[:8]))
        if not os.path.exists(map_path):
            print('map not exist: ', map_path)
            return

        video_map = json.load(open(map_path), encoding='UTF-8')
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
        print('fps, frame_count, width, height:', fps, total_frames, width, height)

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

        # while ret and step < len(select_frame_idx):
        #     if cnt_frame == select_frame_idx[step]:
        #         _, frame = video.retrieve()
        #         bboxes = bbx_extr.extract_bboxes(frame)
        #         bboxes = bboxes.tolist()
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         fea = vf_extractor.extract(frame, bboxes)
        #         fea = fea.squeeze(axis=0).tolist()
        #         # save_path = os.path.join(args.feat_save_dir, video_name)
        #         # if not os.path.exists(save_path):
        #         #     os.makedirs(save_path)
        #         # np.save(os.path.join(save_path, '{:0>8d}.npy'.format(cnt_frame)), fea)
        #         if "feat" in video_map["clips"][select_frame_clip[step]]:
        #             video_map["clips"][select_frame_clip[step]]["feat"].append(fea)
        #         else:
        #             video_map["clips"][select_frame_clip[step]]["feat"] = [fea]
        #
        #         step += 1
        #         print(cnt_frame)
        #
        #     cnt_frame += 1
        #     ret = video.grab()
        # video.release()

        for clip in video_map["clips"]:
            clip["multi_factor"] = {"semantics": None}
            if "feat" in clip:
                clip["multi_factor"]["semantics"] = np.mean(np.array(clip["feat"]), axis=0).tolist()


        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(video_map, fp, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    # python img_feat_extractor.py --frames_dir ./frames --vid_dir /data_share5/douyin/video --vid_csv_path ./vids.csv --feat_save_dir feats
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_path', type=str, default='/innovation_cfs/entertainment/VideoMashup/video')
    parser.add_argument('-dst_path', type=str,
                        default='/innovation_cfs/entertainment/VideoMashup/video_map/transnetv2_duration_frameidx_moviepy_feat')
    parser.add_argument('-map_path', type=str,
                        default='/innovation_cfs/entertainment/VideoMashup/video_map/transnetv2_duration_frameidx_moviepy')
    # parser.add_argument('--frames_dir', type=str, default=None)
    # parser.add_argument('--vid_csv_path', type=str, default=None)
    parser.add_argument('--feat_save_dir', type=str, default=None)
    parser.add_argument('--cfg_file', type=str, default='cfg/test_xyb.yml')
    parser.add_argument('--brivl_checkpoint', type=str,
                        default='/innovation_cfs/mmatch/infguo/weights/BriVL-1.0-5500w.pth')
    parser.add_argument('--bbox_extractor_cfg', type=str,
                        default='bbox_extractor/configs/bua-caffe/extract-bua-caffe-r101.yaml')
    parser.add_argument('-overwrite', default=False, action="store_true")  # whether overwrite the existing results

    args = parser.parse_args()
    abs_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(abs_path, args.cfg_file)
    model_weights = args.brivl_checkpoint


    vf_extractor = ImgFeatureExtractor(cfg_file, model_weights)
    bbx_extr = BboxExtractor(os.path.join(abs_path, args.bbox_extractor_cfg))

    if os.path.isdir(args.src_path):
        for root, _, file_list in os.walk(args.src_path):
            file_list.sort()
            if '周杰伦mv' in root:
                continue
            for file in file_list:
                print('processing: ', file)
                try:
                    video_path = os.path.join(root, file)
                    save_path = os.path.join(args.dst_path, root[len(args.src_path) + 1:])
                    map_path = os.path.join(args.map_path, root[len(args.src_path) + 1:])
                    main(video_path, save_path, map_path, vf_extractor, bbx_extr)
                except Exception as e:
                    traceback.print_exc()
    else:
        main(args.src_path, args.dst_path, args.map_path, vf_extractor, bbx_extr)



