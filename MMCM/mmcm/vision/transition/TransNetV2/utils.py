import os
import pickle
import ffmpeg
import numpy as np
import torch


def get_frames(fn, cache_path, start_frame, end_frame, width=48, height=27):
    '''
    先查询cache_path路径下是否有fn的缓存文件(.pkl),若存在则直接载入, 不存在则通过ffmpeg提取获得
    fn: mp4文件所在路径
    width, height: 转换为指定宽高的视频
    cache_path: 缓存文件所在位置, 例如/data_share7/v_hyggewang/视频单帧数据_h48_w27, 该文件夹中包含有mp4_id.pkl文件则会导入该文件
    return: [总帧率, height, width, 3], np.array
    '''
    cache_mp4_pkl = os.path.join(cache_path, os.path.basename(fn).replace('mp4','pkl'))
    if os.path.exists(cache_mp4_pkl) :
        video = pickle.load(open(cache_mp4_pkl,'rb'))
        assert (video.shape[1]==height and video.shape[2]==width), "mp4缓存文件维度与指定h,w不同"
#         print('mp4帧文件缓存载入成功')
    else:
#         print('使用ffmpeg提取mp4单帧图片数据')
        # 视频转化为np数组
        video_stream, err = (
            ffmpeg
            .input(fn)
            # .filter('select', 'between(n,{},{})'.format(start_frame,end_frame))
            .trim(start_frame=start_frame, end_frame=end_frame)
            .setpts('PTS-STARTPTS')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video

def get_frames_label(mp4_annotations, n_frames, fps):
    '''
    根据mp4的标注信息,以及总帧数长度,给所有帧打标签。
    mp4_annotations: 列表[item...], 每个item是一个字典, 包括有该时间点类别，具体时间。
    fps: 该视频的帧率
    n_frames: 该视频总帧数
    return: many_hot,one_hot维度为(n_frames,)。one_hot表示只有在标注帧的地方是有标签, many_hot表示标注帧的地方前后2帧有相同标签
    '''
    one_hot = np.zeros([n_frames], np.int64)
    many_hot = np.zeros([n_frames], np.int64)
    for item in mp4_annotations:
        if item["class"] == "transition":
            transition_frame_idx = int(item['timaestamp'] * fps) # 根据据标注时间点标注关键帧标签
            one_hot[min(transition_frame_idx, n_frames-1)] = 1   # 卡点帧标记为1 ,有些标注视频总长度低于标注点位，因此需要处理一下
            for i in range(min(transition_frame_idx-2, n_frames-2), min(transition_frame_idx+2, n_frames)):
                many_hot[i] = 1   # 卡点帧周围2帧标记为1(共5帧)
    return one_hot, many_hot

def get_mp4_scenes(frames, one_hot, many_hot):
    """
    对视频按照100帧顺序切片,单个视频顺序切出多段,并且每个片段重叠30帧,例如: 210帧总长度,需要切成0-100帧,70-170帧,140-240帧(重复30帧)
    frames: (T, H, W, 通道数),视频单帧数据.(np.array)
    one_hot: 视频标签one_hot
    many_hot: 视频标签many_hot
    return: 按100帧切割的图片块:[x, 100, H, W, 3], one_hot_scenes:[x, 100], many_hot_scenes:[x, 100], x:表示该mp4被切成了多少块
    """
    frames = torch.from_numpy(frames)
    
    # 重复最后一帧图片使得图片能够正好按规则切分
    if (len(frames)-30)%70 != 0:
        repeat_n = (len(frames)-30)//70*70 + 100 - len(frames)
        last_scenes = frames[-1, :, :, :] # 获取最后一个图像
        last_scenes = torch.unsqueeze(last_scenes, 0).expand([repeat_n, -1, -1, -1])
        frames = torch.cat([frames, last_scenes], dim=0)   # 在原数据最后增加几帧图片
        one_hot = np.concatenate((one_hot, [0]*repeat_n))  # 标签也需要增加长度
        many_hot = np.concatenate((many_hot, [0]*repeat_n))
    
    one_hot = torch.from_numpy(one_hot) # 转化为tensor
    many_hot = torch.from_numpy(many_hot)

    scenes = []         # 用于储存分割的片段
    one_hot_scenes = [] # 用于储存分割片段的label
    many_hot_scenes = []
    # 按规则切片
    for i in range((len(frames)-30)//70):
        scenes.append(frames[i*70:(i*70+100)])
        one_hot_scenes.append(one_hot[i*70:(i*70+100)])
        many_hot_scenes.append(many_hot[i*70:(i*70+100)])
    return torch.stack(scenes, dim=0), torch.stack(one_hot_scenes, dim=0), torch.stack(many_hot_scenes, dim=0), int((len(frames)-30)//70)

def get_mp4_random_scenes(frames, one_hot, many_hot):
    """
    对视频随机设定起点, 抽取100帧数据
    frames: (T, H, W, 通道数),视频单帧数据.(np.array)
    one_hot: 视频标签one_hot
    many_hot: 视频标签many_hot
    return: 随机起点的100帧图片数据[1, 100, H, W, 3], one_hot_scenes:[1, 100], many_hot_scenes:[1, 100]
    """
    number_frames = len(frames)
    start_frames = np.random.randint(0, number_frames-100+1) # 随机获得起始帧位置
    end_frames = start_frames + 100 # 获得终止帧位置
    
    fal_frames = torch.from_numpy(frames[start_frames:end_frames])
    fal_one_hot = torch.from_numpy(one_hot[start_frames:end_frames])
    fal_many_hot = torch.from_numpy(many_hot[start_frames:end_frames])
    return torch.stack([fal_frames], dim=0), torch.stack([fal_one_hot], dim=0), torch.stack([fal_many_hot], dim=0)




