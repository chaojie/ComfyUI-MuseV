# photoalbumn/video2music

本项目为提交给vivo demo的视频和相册配乐中，召回音乐的代码。使用文澜图文匹配技术，视频配乐随机选取5帧图像，相册配乐使用相册中的所有图片，文本为音乐歌词。
每张图片均召回100首歌曲，最后对这100首歌曲的score分数进行排序，输出最终的100首歌曲。热度版本即改变歌曲热度池子范围。


# 使用
### 搭建环境

```
# 环境要求
lmdb==0.99
timm==0.4.12
easydict==1.9
pandas==1.2.4
jsonlines==2.0.0
tqdm==4.60.0
torchvision==0.8.2
numpy==1.20.2
torch==1.7.1
transformers==4.5.1
msgpack_numpy==0.4.7.1
msgpack_python==0.5.6
Pillow==8.3.1
PyYAML==5.4.1
detectron2==0.3+cu102
```

### 视频配乐
```
bash vivo_video2music.sh
```

### 相册配乐
```
bash vivo_photoalbumn2music.sh
```

### 注意
代码需分成独立的3步走。首先提取歌曲歌词，提取完歌词后，提取歌曲和视频特征，最后才进行检索。

