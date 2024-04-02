# -*- encoding: utf-8 -*-
'''
@File    :   text_feat_extractor.py
@Time    :   2021/08/26 10:46:15
@Author  :   Chuhao Jin
@Email   :   jinchuhao@ruc.edu.cn
'''

# here put the import lib

import os
import sys
import pickle
import argparse
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from transformers import AutoTokenizer

from utils import getLanMask
from utils.config import cfg_from_yaml_file, cfg
from models.vl_model import *
from tqdm import tqdm
import pdb

class TextModel(nn.Module):
    def __init__(self, model_cfg):
        super(TextModel, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)

    def forward(self, texts, maskTexts):
        textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        textFea = F.normalize(textFea, p=2, dim=-1)
        return textFea

class TextFeatureExtractor:
    def __init__(self, cfg_file, model_weights, gpu_id = 0):
        self.gpu_id = gpu_id
        self.cfg_file = cfg_file
        self.cfg = cfg_from_yaml_file(self.cfg_file, cfg)
        self.cfg.MODEL.ENCODER = os.path.join(base_dir, self.cfg.MODEL.ENCODER)
        self.text_model = TextModel(model_cfg=self.cfg.MODEL)

        self.text_model = self.text_model.cuda(self.gpu_id)
        model_component = torch.load(model_weights, map_location=torch.device('cuda:{}'.format(self.gpu_id)))
        text_model_component = {}
        for key in model_component["learnable"].keys():
            if "textencoder." in key:
                text_model_component[key] = model_component["learnable"][key]
        self.text_model.learnable.load_state_dict(text_model_component)
        self.text_model.eval()
        
        self.text_transform = AutoTokenizer.from_pretrained('./hfl/chinese-bert-wwm-ext')

    def extract(self, text_input):
        if text_input is None:
            return None
        else:
            text_info = self.text_transform(text_input, padding='max_length', truncation=True,
                                            max_length=self.cfg.MODEL.MAX_TEXT_LEN, return_tensors='pt')
            text = text_info.input_ids.reshape(-1)
            text_len = torch.sum(text_info.attention_mask)
            with torch.no_grad():
                texts = text.unsqueeze(0) 
                text_lens = text_len.unsqueeze(0)
                textMask = getLanMask(text_lens, cfg.MODEL.MAX_TEXT_LEN)
                textMask = textMask.cuda(self.gpu_id)
                texts = texts.cuda(self.gpu_id)
                text_lens = text_lens.cuda(self.gpu_id)
                text_fea = self.text_model(texts, textMask)
                text_fea = text_fea.cpu().numpy()
            return text_fea


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--feat_save_dir', type=str, default=None)
    parser.add_argument('--cfg_file', type=str, default='cfg/test_xyb.yml')
    parser.add_argument('--brivl_checkpoint', type=str, default='/data_share7/sxhong/project/BriVL/weights/BriVL-1.0-5500w.pth')
    args = parser.parse_args()

    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cfg_file)
    model_weights = args.brivl_checkpoint
    save_dir = args.feat_save_dir  
    vfe = TextFeatureExtractor(cfg_file, model_weights)
    if not os.path.exists(args.feat_save_dir):
        os.makedirs(args.feat_save_dir)
    for line in open(args.txt_path):
        try:
            songid, text = line.split(',')[0], line.split('"')[1]
            save_path = os.path.join(save_dir, songid + '.npy')
            if not os.path.exists(save_path):
                text = text.split(',')
                if len(text) >= 5:
                    mid = len(text) // 2
                    text = text[mid-2: mid+2]
                    text_query = ','.join(text)
                fea = vfe.extract(text_query)
                np.save(save_path, fea)
        except:
            pass
            
