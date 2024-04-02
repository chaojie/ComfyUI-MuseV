import os
import numpy as np
import random
from tqdm import tqdm
import argparse
import torch
import sys 
import glob
import csv
import torch.utils.data as data
import pandas as pd
import re
import pdb
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/'+'..')) 

parser = argparse.ArgumentParser()
parser.add_argument('--vid_csv_path', type=str, default=None)
parser.add_argument('--image_dir', type=str, default=None)
parser.add_argument('--text_dir', type=str, default=None)
parser.add_argument('--lyric_txt_path', type=str, default=None)
parser.add_argument('--save_csv', type=str, default=None)
parser.add_argument('--gpu', type=str, default='3')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# python video2img_retrieval.py --vid_csv_path ../vids.csv --image_dir ../feats --text_dir ../../BriVL-git/BriVL_code_inference/feat/text_vivo_5w_songids/ --lyric_txt_path  /data/home/sxhong/tools/get_lyric/data/vivo_top5w_songid_lyrics.lyric --save_csv test.csv


def get_ImgFeat(img_feats):
    # 获取imgs_feat
    img_dict = {}
    for i, np_img_path in enumerate(img_feats):
        img_dict[i] = np_img_path
        if i == 0:
            np_img = np.expand_dims(np.load(np_img_path).astype(np.float64), axis=0)
        else:
            np_img = np.concatenate((np_img, np.expand_dims(np.load(np_img_path).astype(np.float64), axis=0)), axis=0)
    img = torch.from_numpy(np_img).cuda()
    return img_dict, img

      
class Text_Dataset(data.Dataset):
    def __init__(self, text_dir):
        self.text_feats = glob.glob(os.path.join(text_dir, '*.npy'))
        
    def __len__(self):
        return len(self.text_feats)
    
    def __getitem__(self, index):
        text_path = self.text_feats[index]
        songid = text_path.split('/')[-1].split('.')[0]
        text_feat = np.load(text_path)
        return songid, text_feat

def get_TextFeat(text_dir):
    # pdb.set_trace()
    # 获取texts_feat 
    dataset = Text_Dataset(text_dir)
    dataloader = data.DataLoader(dataset, batch_size=5000, shuffle=False)
    all_songids = []
    for i, (songid_, text_feats) in enumerate(dataloader):
        all_songids.extend(list(songid_))
        if i == 0:
            text = text_feats
        else:
            text = torch.cat((text, text_feats), 0)
    text = text.squeeze(dim=1)
    text = text.cuda()
    return all_songids, text


def get_lyric(lyric_txt_path):
    # songid对应的歌词
    text_dict = {}
    for line in open(lyric_txt_path):
        try:
            songid, text_query = line.split(',')[0], line.split('"')[1]
            text_dict[songid] = text_query
        except:
            pass
    return text_dict


vids = pd.read_csv(args.vid_csv_path, dtype=str)['vid'].to_list()
all_songids, text = get_TextFeat(args.text_dir)
text_dict = get_lyric(args.lyric_txt_path)
vid2songid = {}

for vid in vids:    
    img_feat_paths = glob.glob(os.path.join(args.image_dir, str(vid)+'_'+'*.npy')) 
    img_dict, img = get_ImgFeat(img_feat_paths)
    N_img = img.shape[0]
    N_text = text.shape[0]
    scores = torch.zeros((N_img, N_text), dtype=torch.float32).cuda()

    print('Pair-to-pair: calculating scores')
    for i in tqdm(range(N_img)): # row: image  col: text
        scores[i, :] = torch.sum(img[i] * text, -1)
    
    songid2score = {}
    songids = []
    songids2scores = []
    for i, score in enumerate(scores):
        indices = torch.argsort(score, descending=True)
        songids = []
        for idx in indices:
            if len(songids) <= 80:
                idx = int(idx.cpu().numpy())
                query_text = text_dict[all_songids[idx]]
                query_text = query_text.split(',')
                query_text = (query_text[len(query_text) // 2]).replace(' ','').replace('*','')
                for exist_songid in songids:
                    key_text = (text_dict[exist_songid]).replace(' ','').replace('*','')
                    if re.findall(query_text, key_text):
                        if float(songid2score[exist_songid]) < float(score[idx].cpu().numpy()):
                            songid2score.pop(exist_songid)
                            songid2score[all_songids[idx]] = str(score[idx].cpu().numpy())
                            break
                
                songid2score[all_songids[idx]] = str(score[idx].cpu().numpy())
            else:
                break
    sorted_songid2score = sorted(songid2score.items(), key=lambda x:float(x[1]), reverse=True)
    select_songids = ', '.join([songid for songid, score in sorted_songid2score[:100]])
    correspond_scores = ', '.join([score for songid, score in sorted_songid2score[:100]])
    vid2songid[vid] = (select_songids, correspond_scores)

data = []
for vid, values in vid2songid.items():
    songids, scores = values
    data.append([vid, songids, scores])

df = pd.DataFrame(data, columns=['img', 'songids', 'scores'])
df.to_csv(args.save_csv, index=False)