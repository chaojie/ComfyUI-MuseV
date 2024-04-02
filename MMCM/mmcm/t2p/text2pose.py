#from __future__ import absolute_import
import sys
import io
import os
sys.argv = ['GPT_eval_multi.py']

# 将项目根目录添加到sys.path中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, PROJECT_ROOT)
CKPT_ROOT="/cfs-datasets/public_models/motion"

from .options import option_transformer as option_trans

import sys
print(sys.path[0])

import clip
import torch
import cv2
import numpy as np
from  .models import vqvae as vqvae
from  .models import t2m_trans as trans
import warnings
from  .visualization import plot_3d_global as plot_3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import time
import random


warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from math import cos,sin,radians

args = option_trans.get_args_parser()

args.dataname = 't2m'
args.resume_pth = os.path.join(CKPT_ROOT,'pretrained/VQVAE/net_last.pth')
args.resume_trans = os.path.join(CKPT_ROOT,'pretrained/VQTransformer_corruption05/net_best_fid.pth')
args.down_t = 2
args.depth = 3
args.block_size = 51

def replace_space_with_underscore(s):
    return s.replace(' ', '_')


def Rz(angle):
  theta=radians(angle)
  return np.array([[cos(theta), -sin(theta), 0],
             [sin(theta), cos(theta),  0],
             [0,          0,           1]])


def Rx(angle):
  theta=radians(angle)
  return np.array(
    [[1,   0,         0],
    [0 , cos(theta), -sin(theta)],
    [0,  sin(theta), cos(theta)]])

def generate_cuid():
    timestamp = hex(int(time.time() * 1000))[2:]
    random_str = hex(random.randint(0, 0xfffff))[2:]
    return (timestamp + random_str).zfill(10)

def smpl_to_openpose18(smpl_keypoints):
    '''
    22关键点SMPL对应关系解释 
    [0, 2, 5, 8, 11]
    这个列表表示SMPL模型中左腿的连接方式，从骨盆（0号关键点）开始，连接左大腿（2号关键点）、左小腿（5号关键点）、左脚（8号关键点）和左脚尖（11号关键点）。
    
    [0, 1, 4, 7, 10]
    这个列表表示SMPL模型中右腿的连接方式，从骨盆（0号关键点）开始，连接右大腿（1号关键点）、右小腿（4号关键点）、右脚（7号关键点）和右脚尖（10号关键点）。
    
    [0, 3, 6, 9, 12, 15]
    这个列表表示SMPL模型中躯干的连接方式，从骨盆（0号关键点）开始，连接脊柱（3号关键点）、颈部（6号关键点）、头部（9号关键点）、左肩膀（12号关键点）、右肩膀（15号关键点）。
    
    [9, 14, 17, 19, 21]
    这个列表表示SMPL模型中左臂的连接方式，从左肩膀（9号关键点）开始，连接左上臂（14号关键点）、左前臂（17号关键点）、左手腕（19号关键点）和左手（21号关键点）。
    
    [9, 13, 16, 18, 20]
    这个列表表示SMPL模型中右臂的连接方式，从右肩膀（9号关键点）开始，连接右上臂（13号关键点）、右前臂（16号关键点）、右手腕（18号关键点）和右手（20号关键点）。
    
    目前转Openpose忽略掉了SMPL的肩膀关键点
    '''
    openpose_keypoints = np.zeros((18, 3))
    openpose_keypoints[0] = smpl_keypoints[9] # nose
    openpose_keypoints[0][1] = openpose_keypoints[0][1]+0.3 # 


    openpose_keypoints[1] = smpl_keypoints[6] # neck
    openpose_keypoints[2] = smpl_keypoints[16] # right shoulder 
    openpose_keypoints[3] = smpl_keypoints[18] # right elbow
    openpose_keypoints[4] = smpl_keypoints[20] # right wrist
    openpose_keypoints[5] = smpl_keypoints[17] # left shoulder
    openpose_keypoints[6] = smpl_keypoints[19] # left elbow
    openpose_keypoints[7] = smpl_keypoints[21] # left wrist

    #TODO: Experiment,将neck的关键点抬高&&将nose的关键点相对高度关系与neck保持一致
    openpose_keypoints[1][0]=(openpose_keypoints[2][0]+openpose_keypoints[5][0])/2
    openpose_keypoints[1][1]=(openpose_keypoints[2][1]+openpose_keypoints[5][1])/2
    openpose_keypoints[1][2]=(openpose_keypoints[2][2]+openpose_keypoints[5][2])/2
    openpose_keypoints[0][1] = openpose_keypoints[1][1]+0.3 # 


    openpose_keypoints[8] = smpl_keypoints[1] # right hip
    openpose_keypoints[9] = smpl_keypoints[4] # right knee
    openpose_keypoints[10] = smpl_keypoints[7] # right ankle
    openpose_keypoints[11] = smpl_keypoints[2] # left hip
    openpose_keypoints[12] = smpl_keypoints[5] # left knee
    openpose_keypoints[13] = smpl_keypoints[8] # left ankle

    #TODO: Experiment,手工指定脸部关键点测试是否能够指定身体朝向
    #openpose_keypoints[0][0] = openpose_keypoints[0][0]+0.3#测试0坐标轴方向(水平向右)
    #openpose_keypoints[0][2] = openpose_keypoints[0][2]#测试2坐标轴方向（向外
    #openpose_keypoints[0][1] = openpose_keypoints[0][1]+0.5#测试1坐标轴方向（垂直向上
    openpose_keypoints[14] = openpose_keypoints[0] # right eye
    openpose_keypoints[14][1]=openpose_keypoints[14][1]+0.05
    openpose_keypoints[14][0]=openpose_keypoints[14][0]+0.3*(openpose_keypoints[2][0]-openpose_keypoints[1][0])
    openpose_keypoints[14][2]=openpose_keypoints[14][2]+0.3*(openpose_keypoints[2][2]-openpose_keypoints[1][2])

    openpose_keypoints[15] = openpose_keypoints[0] # left eye
    openpose_keypoints[15][1]=openpose_keypoints[15][1]+0.05
    openpose_keypoints[15][0]=openpose_keypoints[15][0]+0.3*(openpose_keypoints[5][0]-openpose_keypoints[1][0])
    openpose_keypoints[15][2]=openpose_keypoints[15][2]+0.3*(openpose_keypoints[5][2]-openpose_keypoints[1][2])
    
    openpose_keypoints[16] = openpose_keypoints[0] # right ear
    openpose_keypoints[16][0]=openpose_keypoints[16][0]+0.7*(openpose_keypoints[2][0]-openpose_keypoints[1][0])
    openpose_keypoints[16][2]=openpose_keypoints[16][2]+0.7*(openpose_keypoints[2][2]-openpose_keypoints[1][2])    
    
    openpose_keypoints[17] = openpose_keypoints[0] # left ear
    openpose_keypoints[17][0]=openpose_keypoints[17][0]+0.7*(openpose_keypoints[5][0]-openpose_keypoints[1][0])
    openpose_keypoints[17][2]=openpose_keypoints[17][2]+0.7*(openpose_keypoints[5][2]-openpose_keypoints[1][2])    
    
    return openpose_keypoints






# TODO: debug only, need to be deleted before unload
## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root=CKPT_ROOT)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print("loaded CLIP model")
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                embed_dim=1024,
                                clip_dim=args.clip_dim,
                                block_size=args.block_size,
                                num_layers=9,
                                n_head=16,
                                drop_out_rate=args.drop_out_rate,
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()

mean = torch.from_numpy(np.load(os.path.join(CKPT_ROOT,'./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy'))).cuda()
std = torch.from_numpy(np.load(os.path.join(CKPT_ROOT,'./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy'))).cuda()



def get_open_pose(text,height,width,save_path,video_length):
    CKPT_ROOT = os.path.dirname(os.path.abspath(__file__))

    clip_text=[text]
    print(f"Motion Prompt: {text}")
    # cuid=generate_cuid()
    # print(f"Motion Generation cuid: {cuid}")

    # clip_text = ["the person jump and spin twice,then running straght and sit down. "]  #支持单个token的生成

    # change the text here



    text = clip.tokenize(clip_text, truncate=False).cuda()
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)

    from utils.motion_process import recover_from_ric
    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22) 
    xyz = pred_xyz.reshape(1, -1, 22, 3) 

    np.save('motion.npy', xyz.detach().cpu().numpy())


    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(),clip_text, ['smpl.gif'])

    res=xyz.detach().cpu().numpy()
    points_3d_list=res[0]
    frame_num=points_3d_list.shape[0]

    open_pose_list=np.array(points_3d_list)
    print("The total SMPL sequence shape is : "+str(open_pose_list.shape))

    max_val = np.max(open_pose_list, axis=(0, 1))
    min_val = np.min(open_pose_list, axis=(0, 1))

    print("三维坐标在坐标系上的最大值：", max_val)
    print("三维坐标在坐标系上的最小值：", min_val)


    check= smpl_to_openpose18(open_pose_list[0]) # 18个关键点
    print("********SMPL_2_OpenPose_List(14/18)********")
    print(check)
    print("*************************")
    print(f"Total Frame Number: {frame_num}")
    img_list=[]
    for step in tqdm(range(0,frame_num)):
        # 生成图像
        dpi=84
        fig =plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        limits=2

        ax.set_xlim(-limits*0.7, limits*0.7)
        ax.set_ylim(0, limits*1.5)#上下
        ax.set_zlim(0, limits*1.5)# 前后
        ax.grid(b=False)
        #ax.dist = 1
        ax.set_box_aspect([1.4, 1.5, 1.5],zoom=3.5)#  坐标轴比例 TODO:这个比例可能有问题，会出现超出坐标范围的bug

        # 关键点坐标，每行包含(x, y, z)
        keypoints = smpl_to_openpose18(open_pose_list[step]) # 18个关键点

        # 运动学链 目前只用到body部分
        kinematic_chain = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)]
        #kinematic_chain = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]

        # 颜色RGB

        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 192, 203), (0, 165, 255), (19, 69, 139), (173, 216, 230), (34, 139, 34), (0, 0, 128), (184, 134, 11), (139, 0, 139), (0, 100, 0), (0, 255, 255), (0, 255, 0), (216, 191, 216), (255, 255, 224)]
        #colors=[(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 192, 203), (0, 165, 255), (19, 69, 139), (173, 216, 230), (34, 139, 34), (0, 0, 128), (184, 134, 11), (139, 0, 139), (0, 100, 0)]
        
        #18点
        joint_colors=[(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),(255,0,0)]
        #14点主干
        #joint_colors=[(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255)]
        #运动链连线是joint颜色的60%
        
        
        #plt颜色在0-1之间
        rgb_color2=[]
        joint_rgb_color2=[]
        kinematic_chain_rgb_color2=[]
        for color in joint_colors:
            joint_rgb_color2.append(tuple([x/255 for x in color]))
            kinematic_chain_rgb_color2.append(tuple([x*0.6/255 for x in color]))    #运动链连线是joint颜色的60%

        # 可视化结果
        for i in range(0,18):
            # 绘制关键点
            ax.scatter(keypoints[i][0], keypoints[i][1], keypoints[i][2], s=50, c=joint_rgb_color2[i], marker='o')

            # 绘制运动学链
            for j in range(len(kinematic_chain)):
                if kinematic_chain[j][1] == i:
                    ax.plot([keypoints[kinematic_chain[j][0]][0], keypoints[kinematic_chain[j][1]][0]], [keypoints[kinematic_chain[j][0]][1], keypoints[kinematic_chain[j][1]][1]], [keypoints[kinematic_chain[j][0]][2], keypoints[kinematic_chain[j][1]][2]], c=kinematic_chain_rgb_color2[i], linewidth=5)

        # 调整视角
        ax.view_init(elev=110, azim=-90)
        plt.axis('off')

        
        # 保存图片
        # 将图像数据输出为图像数组
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_tmp_path=str(f"{save_path}/{str(step)}.jpg")
        plt.savefig(os.path.join(CKPT_ROOT,image_tmp_path))#RGB
        img=cv2.imread(os.path.join(CKPT_ROOT,image_tmp_path))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_list.append(img)
    res=[]
    if len(img_list)>=video_length:
        key_frame_sample_step=int(len(img_list)/video_length)
    else:
        print("ERROR: video length is too long")
        key_frame_sample_step=1

    for i in range(0,len(img_list),key_frame_sample_step):
        res.append(img_list[i])
    
    return res



def offline_get_open_pose(text,motion_text,height,width,save_path):
    #motion_text=text

    clip_text=[text]
    print(f"Motion Prompt: {text}")
    cuid=generate_cuid()
    print(f"Motion Generation cuid: {cuid}")

    # clip_text = ["the person jump and spin twice,then running straght and sit down. "]  #支持单个token的生成

    # change the text here



    text = clip.tokenize(clip_text, truncate=False).cuda()
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)

    from utils.motion_process import recover_from_ric
    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22) 
    xyz = pred_xyz.reshape(1, -1, 22, 3) 
    res=xyz.detach().cpu().numpy()
    np.save(f'{save_path}/{replace_space_with_underscore(motion_text)}.npy', res)


    pose_vis = plot_3d.draw_to_batch(res,clip_text, ['smpl.gif'])
    



if __name__ == "__main__":

    text="walk around, jump, run straght."
    pose = get_open_pose(text,512,512)
    #pdb.set_trace()

