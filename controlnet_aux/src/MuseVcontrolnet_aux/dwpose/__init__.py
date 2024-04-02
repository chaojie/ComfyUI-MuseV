# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from . import util
from pprint import pprint


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas
def draw_pose_on_canvas(pose, canvas):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas


def candidate2pose(candidate, subset):
    nums, keys, locs = candidate.shape
    body = candidate[:,:18].copy()
    body = body.reshape(nums*18, locs)
    score = subset[:,:18]
    
    for i in range(len(score)):
        for j in range(len(score[i])):
            if score[i][j] > 0.3:
                score[i][j] = int(18*i+j)
            else:
                score[i][j] = -1

    un_visible = subset<0.3
    candidate[un_visible] = -1

    foot = candidate[:,18:24]

    faces = candidate[:,24:92]

    hands = candidate[:,92:113]
    hands = np.vstack([hands, candidate[:,113:]])
    
    bodies = dict(candidate=body, subset=score)
    pose = dict(bodies=bodies, hands=hands, faces=faces)
    return pose

def size_calculate(H,W, resolution):
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H,W
def pose2map(pose,H_in,W_in,detect_resolution,image_resolution):
    H,W = size_calculate(H_in, W_in, detect_resolution)
    detected_map = draw_pose(pose, H, W)
    detected_map = HWC3(detected_map)
    
    H,W = size_calculate(H,W,image_resolution)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        
    return detected_map

def pose2map_on_canvas(pose,H_in,W_in,detect_resolution,image_resolution,canvas):
    H,W = size_calculate(H_in, W_in, detect_resolution)
    detected_map = draw_pose_on_canvas(pose, canvas)
    detected_map = HWC3(detected_map)
    
    H,W = size_calculate(H,W,image_resolution)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        
    return detected_map


class DWposeDetector:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu"):
        from .wholebody import Wholebody

        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, device)
    
    def to(self, device):
        self.pose_estimation.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", return_pose_dict=False, return_pose_only=False, **kwargs):
        """
        Args:
            return_pose_only (bool): if true, only return pose keypoints in array format
            return_pose_dict (bool): if true, return 1) pose image; 2) pose keypoints in dict format
            
        """
        
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        with torch.no_grad():
            # print('=========== in controlnet_aux dwpose')
            candidate, subset = self.pose_estimation(input_image)
            # print(candidate.shape)
            # print(subset.shape)
            # candidate shape (1, 134, 2)
            # subset          (1, 134)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)

            if return_pose_only:
                return (candidate, subset)


            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)
            
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)
                
            if return_pose_dict:
                return detected_map, pose
            else:
                return detected_map



# pip install -U openmim &&  mim install mmengine   "mmcv>=2.0.1"  "mmdet>=3.1.0"  "mmpose>=1.1.0"