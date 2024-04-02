# -*- coding:UTF8 -*-

"""Image bounding-box extraction process."""

import os
import sys
import sys
import cv2
import numpy as np
import glob

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.engine import DefaultTrainer
from detectron2.layers.nms import nms

from .bbox_utils.extract_utils import get_image_blob
from .bbox_models import add_config
from .bbox_models.bua.box_regression import BUABoxes
from .bbox_models.bua import add_bottom_up_attention_config

class BboxExtractor:
    def __init__(self, cfg_file, gpu_id = 0):
        
        self.cfg_file = cfg_file
        self.gpu_id = gpu_id
        self.cfg = get_cfg()
        add_bottom_up_attention_config(self.cfg, True)
        self.cfg.merge_from_file(self.cfg_file)
        self.cfg.freeze()
        default_setup(self.cfg, None)

        self.bbox_extract_model = DefaultTrainer.build_model(self.cfg)
#        self.bbox_extract_model.cuda(gpu_id)
        bbox_extract_model_dict = self.bbox_extract_model.state_dict()
        bbox_extract_checkpoint_dict = torch.load(self.cfg.MODEL.WEIGHTS, map_location=torch.device('cuda:0'))['model']
        bbox_extract_checkpoint_dict = {k:v for k, v in bbox_extract_checkpoint_dict.items() if k in bbox_extract_model_dict}
        bbox_extract_model_dict.update(bbox_extract_checkpoint_dict)
        self.bbox_extract_model.load_state_dict(bbox_extract_model_dict)
        # self.bbox_extract_model = torch.nn.DataParallel(self.bbox_extract_model, device_ids=self.gpus)
        self.bbox_extract_model.eval()

    def clean_bbox(self, dataset_dict, boxes, scores):
        MIN_BOXES = self.cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
        MAX_BOXES = self.cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = self.cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

        scores = scores[0]
        boxes = boxes[0]
        num_classes = scores.shape[1]
        boxes = BUABoxes(boxes.reshape(-1, 4))
        boxes.clip((dataset_dict['image'].shape[1]/dataset_dict['im_scale'], dataset_dict['image'].shape[2]/dataset_dict['im_scale']))
        boxes = boxes.tensor.view(-1, num_classes*4)  # R x C x 4

        cls_boxes = torch.zeros((boxes.shape[0], 4))
        for idx in range(boxes.shape[0]):
            cls_idx = torch.argmax(scores[idx, 1:]) + 1
            cls_boxes[idx, :] = boxes[idx, cls_idx * 4:(cls_idx + 1) * 4]

        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, num_classes):
                cls_scores = scores[:, cls_ind]
                keep = nms(cls_boxes, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                                 cls_scores[keep],
                                                 max_conf[keep])
                
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        image_bboxes = cls_boxes[keep_boxes]

        return image_bboxes

    def extract_bboxes(self, img_path):
        if type(img_path)==str:
            im = cv2.imread(img_path)
        else:
            im = img_path
        if im is None:
            print("img is None!")
            return None
        else:
            dataset_dict = get_image_blob(im, self.cfg.MODEL.PIXEL_MEAN)
            with torch.set_grad_enabled(False):
                boxes, scores = self.bbox_extract_model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            boxes = self.clean_bbox(dataset_dict, boxes, scores)

            return boxes # boxes type tensor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, dest="img_path", default="/data1/sxhong/video_image")
    parser.add_argument('--out_path', type=str, dest="output_file", default="./data/shuishoupai_bbox")
    args = parser.parse_args()
    image_paths = glob.glob(os.path.join(args.img_path, '*'))
    
    for image_path in image_paths:
        save_file = os.path.join(args.output_file, image_path.split('/')[-1].split('.')[0] + '.npz')
        
        print(save_file)
        abs_path = os.path.dirname(os.path.abspath(__file__))
        bbx_extr = BboxExtractor(os.path.join(abs_path, 'configs/bua-caffe/extract-bua-caffe-r101.yaml'))
        bboxes = bbx_extr.extract_bboxes(image_path)
        np.savez_compressed(save_file, bbox=bboxes)
        print(code)
        np_bbox = bboxes.numpy().astype(np.int32)
#         print(np_bbox)
        print(bboxes.shape)

