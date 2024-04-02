import random

import torch
from torch import nn
import torch.nn.functional as F
import ffmpeg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from .utils import get_frames


class TransNetV2(nn.Module):
    def __init__(self, F=16, L=3, S=2, D=1024):
        super(TransNetV2, self).__init__()
        self.SDDCNN = nn.ModuleList(
            [
                StackedDDCNNV2(
                    in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0
                )
            ]
            + [
                StackedDDCNNV2(
                    in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2**i
                )
                for i in range(1, L)
            ]
        )

        # 帧相似网络
        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2**i) * 4 for i in range(L)]),
            lookup_window=101,
            output_dim=128,
            similarity_dim=128,
            use_bias=True,
        )

        # 颜色相似网络
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128)

        # dropout
        self.dropout = nn.Dropout(0.5)

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  #
        output_dim = output_dim + 128  # 使用了帧相似网络, 维度需要加128
        output_dim = output_dim + 128  # 使用了颜色相似网络, 维度需要再加128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)

    def forward(self, inputs):
        # 输入必须为torch.uint8, (h,w)=(27,48)的图片batch样本
        #         assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, "incorrect input type and/or shape"

        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        with torch.autograd.set_detect_anomaly(True):
            x = inputs.permute([0, 4, 1, 2, 3]).float()
            x = x.div_(255.0)

            # 收集每一层的SDDCNN特征图
            block_features = []
            for block in self.SDDCNN:
                x = block(x)
                block_features.append(x)

            x = x.permute(0, 2, 3, 4, 1)  # 把维度从[B, 通道数, T, H, W] 转化为 [B, T, H, W, 通道数]
            x = x.reshape(x.shape[0], x.shape[1], -1)

            x = torch.cat(
                [self.frame_sim_layer(block_features), x], 2
            )  # 在最后一维度cat上block_features输出的特征
            x = torch.cat(
                [self.color_hist_layer(inputs), x], 2
            )  # 在最后一维度cat上color_hist_layer输出的特征

            x = F.relu(self.fc1(x))
            x = self.dropout(x)

            one_hot = self.cls_layer1(x)
            many_hot = self.cls_layer2(x)
            return one_hot, many_hot

    # 预测MP4文件转换帧，并给出对应帧位置
    def predict_video(
        self,
        mp4_file,
        cache_path="",
        c_box=None,
        width=48,
        height=27,
        input_frames=100,
        overlap=30,
        sample_fps=30,
        threshold=0.3,
    ):
        """
        mp4_file: ~/6712566330782010632.mp4
        cache_path: ~/视频单帧数据_h48_w27
        return: [x,x,...] 点位时间
        """
        assert overlap % 2 == 0
        assert input_frames > overlap
        # fps = eval(ffmpeg.probe(mp4_file)['streams'][0]['r_frame_rate']) # 获取视频的视频帧率
        # total_frames = int(ffmpeg.probe(mp4_file)['streams'][0]['nb_frames']) # 获取视频的总帧数
        # duration = float(ffmpeg.probe(mp4_file)['streams'][0]['duration']) # 获取视频的总时长
        video = VideoFileClip(mp4_file)
        # video = video.subclip(0, 60 * 10)
        fps = video.fps
        duration = video.duration
        total_frames = int(duration * fps)
        w, h = video.size
        print(fps, duration, total_frames, w, h)

        if c_box:
            video.crop(*c_box)

        frame_iter = video.iter_frames(fps=sample_fps)
        sample_total_frames = int(sample_fps * duration)
        frame_list = []
        for i in range(sample_total_frames // (input_frames - overlap) + 1):
            # if i==1:
            #     break
            frame_list = frame_list[-overlap:]
            start_frame = i * (input_frames - overlap)
            end_frame = min(start_frame + input_frames, sample_total_frames)
            print("start_frame & end_frame: ", start_frame, end_frame)
            for frame in frame_iter:
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_list.append(frame)
                if len(frame_list) == end_frame - start_frame:
                    break
            frames = torch.Tensor(frame_list)  # 获得帧
            if frames.shape[0] < end_frame - start_frame:
                # 原视频的视频时长比音频时长短，体现出来的是原视频最后有声音没画面
                print(
                    "total_frames is wrong: ",
                    total_frames,
                    "-->",
                    start_frame + frames.shape[0],
                )
                # sample_total_frames = start_frame + frames.shape[0]
                # fps = total_frames / duration
            frames = frames.cuda()
            # single_frame_pred和all_frame_pred都是输出window_size长的是否转场概率，
            single_frame_pred, all_frame_pred = self.forward(
                frames.unsqueeze(0)
            )  # 前向推理
            # single_frame_pred = F.softmax(single_frame_pred, dim=-1) # 获得每一帧对应的类别概率
            # single_frame_pred = torch.argmax(single_frame_pred, dim=-1).reshape(-1)
            single_frame_pred = torch.sigmoid(single_frame_pred).reshape(-1)
            all_frame_pred = torch.sigmoid(all_frame_pred).reshape(-1)

            # single_frame_pred = (single_frame_pred>threshold)*1
            if total_frames > end_frame:
                if i == 0:
                    single_frame_pred_label = single_frame_pred[: -overlap // 2]
                    all_frame_pred_label = all_frame_pred[: -overlap // 2]
                else:
                    single_frame_pred_label = torch.cat(
                        (
                            single_frame_pred_label,
                            single_frame_pred[overlap // 2 : -overlap // 2],
                        ),
                        dim=0,
                    )
                    all_frame_pred_label = torch.cat(
                        (
                            all_frame_pred_label,
                            all_frame_pred[overlap // 2 : -overlap // 2],
                        ),
                        dim=0,
                    )
            else:
                if i == 0:
                    single_frame_pred_label = single_frame_pred
                    all_frame_pred_label = all_frame_pred
                else:
                    single_frame_pred_label = torch.cat(
                        (single_frame_pred_label, single_frame_pred[overlap // 2 :]),
                        dim=0,
                    )
                    all_frame_pred_label = torch.cat(
                        (all_frame_pred_label, all_frame_pred[overlap // 2 :]), dim=0
                    )
                break

        single_frame_pred_label = single_frame_pred_label.cpu().numpy()
        all_frame_pred_label = all_frame_pred_label.cpu().numpy()

        return (
            single_frame_pred_label,
            all_frame_pred_label,
            fps,
            total_frames,
            duration,
            h,
            w,
        )

        # transition_index = torch.where(pred_label==1)[0].cpu().numpy() # 转场帧位置
        # transition_index = transition_index.astype(np.float)
        # # 对返回结果做后处理合并相邻帧
        # result_transition = []
        # for i, transition in enumerate(transition_index):
        #     if i == 0:
        #         result_transition.append([transition])
        #     else:
        #         if abs(result_transition[-1][-1]-transition) == 1:
        #             result_transition[-1].append(transition)
        #         else:
        #             result_transition.append([transition])
        #
        # result_transition = [[0]] + [[item[0], item[-1]] if len(item)>1 else [item[0]] for item in result_transition] + [[total_frames]]
        #
        # return result_transition, fps, total_frames, duration, h, w

    def predict_video_2(
        self,
        mp4_file,
        cache_path="",
        c_box=None,
        width=48,
        height=27,
        input_frames=100,
        overlap=30,
        sample_fps=30,
        threshold=0.3,
    ):
        """
        mp4_file: ~/6712566330782010632.mp4
        cache_path: ~/视频单帧数据_h48_w27
        return: [x,x,...] 点位时间
        """
        assert overlap % 2 == 0
        assert input_frames > overlap
        # fps = eval(ffmpeg.probe(mp4_file)['streams'][0]['r_frame_rate']) # 获取视频的视频帧率
        # total_frames = int(ffmpeg.probe(mp4_file)['streams'][0]['nb_frames']) # 获取视频的总帧数
        # duration = float(ffmpeg.probe(mp4_file)['streams'][0]['duration']) # 获取视频的总时长
        video = VideoFileClip(mp4_file)
        # video = video.subclip(0, 60 * 10)
        fps = video.fps
        duration = video.duration
        total_frames = int(duration * fps)
        w, h = video.size
        print(fps, duration, total_frames, w, h)

        if c_box:
            video.crop(*c_box)

        frame_iter = video.iter_frames(fps=sample_fps)
        sample_total_frames = int(sample_fps * duration)
        frame_list = []
        for i in range(sample_total_frames // (input_frames - overlap) + 1):
            # if i==1:
            #     break
            frame_list = frame_list[-overlap:]
            start_frame = i * (input_frames - overlap)
            end_frame = min(start_frame + input_frames, sample_total_frames)
            print("start_frame & end_frame: ", start_frame, end_frame)
            for frame in frame_iter:
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_list.append(frame)
                if len(frame_list) == end_frame - start_frame:
                    break
            frames = torch.Tensor(frame_list)  # 获得帧
            if frames.shape[0] < end_frame - start_frame:
                # 原视频的视频时长比音频时长短，体现出来的是原视频最后有声音没画面
                print(
                    "total_frames is wrong: ",
                    total_frames,
                    "-->",
                    start_frame + frames.shape[0],
                )
                # sample_total_frames = start_frame + frames.shape[0]
                # fps = total_frames / duration
            frames = frames.cuda()
            single_frame_pred, all_frame_pred = self.forward(
                frames.unsqueeze(0)
            )  # 前向推理
            # single_frame_pred = F.softmax(single_frame_pred, dim=-1) # 获得每一帧对应的类别概率
            # single_frame_pred = torch.argmax(single_frame_pred, dim=-1).reshape(-1)
            single_frame_pred = torch.sigmoid(single_frame_pred).reshape(-1)
            all_frame_pred = torch.sigmoid(all_frame_pred).reshape(-1)

            # single_frame_pred = (single_frame_pred>threshold)*1
            if total_frames > end_frame:
                if i == 0:
                    single_frame_pred_label = single_frame_pred[: -overlap // 2]
                    all_frame_pred_label = all_frame_pred[: -overlap // 2]
                else:
                    single_frame_pred_label = torch.cat(
                        (
                            single_frame_pred_label,
                            single_frame_pred[overlap // 2 : -overlap // 2],
                        ),
                        dim=0,
                    )
                    all_frame_pred_label = torch.cat(
                        (
                            all_frame_pred_label,
                            all_frame_pred[overlap // 2 : -overlap // 2],
                        ),
                        dim=0,
                    )
            else:
                if i == 0:
                    single_frame_pred_label = single_frame_pred
                    all_frame_pred_label = all_frame_pred
                else:
                    single_frame_pred_label = torch.cat(
                        (single_frame_pred_label, single_frame_pred[overlap // 2 :]),
                        dim=0,
                    )
                    all_frame_pred_label = torch.cat(
                        (all_frame_pred_label, all_frame_pred[overlap // 2 :]), dim=0
                    )
                break

        single_frame_pred_label = single_frame_pred_label.cpu().numpy()
        all_frame_pred_label = all_frame_pred_label.cpu().numpy()

        return (
            single_frame_pred_label,
            all_frame_pred_label,
            fps,
            total_frames,
            duration,
            h,
            w,
        )


class StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters,
        n_blocks,
        filters,
        shortcut=True,
        pool_type="avg",
        stochastic_depth_drop_prob=0.0,
    ):
        super(StackedDDCNNV2, self).__init__()

        self.shortcut = shortcut
        # 定义DDCNN层
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=F.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )  # 有n_blocks层数量的DilateDCNNV2模块

        # 定义pool层
        self.pool = (
            nn.MaxPool3d(kernel_size=(1, 2, 2))
            if pool_type == "max"
            else nn.AvgPool3d(kernel_size=(1, 2, 2))
        )
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        # DDCNN层前向传播
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:  # 记录第一层的结果作为残差连接
                shortcut = x

        x = F.relu(x)
        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x = x + shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None):
        super(DilatedDCNNV2, self).__init__()

        self.Conv3D_1 = Conv3DConfigurable(
            in_filters, filters, 1, use_bias=not batch_norm
        )
        self.Conv3D_2 = Conv3DConfigurable(
            in_filters, filters, 2, use_bias=not batch_norm
        )
        self.Conv3D_4 = Conv3DConfigurable(
            in_filters, filters, 4, use_bias=not batch_norm
        )
        self.Conv3D_8 = Conv3DConfigurable(
            in_filters, filters, 8, use_bias=not batch_norm
        )

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation  # 激活函数定义

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(
        self, in_filters, filters, dilation_rate, separable=True, use_bias=True
    ):
        super(Conv3DConfigurable, self).__init__()

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters,
                2 * filters,
                kernel_size=(1, 3, 3),
                dilation=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


# 帧相似网络构建
class FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_filters,
        similarity_dim=128,
        lookup_window=101,
        output_dim=128,
        use_bias=False,
    ):
        super(FrameSimilarity, self).__init__()

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = F.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(
            x, x.transpose(1, 2)
        )  # [batch_size, time_window, time_window]余弦相似度
        similarities_padded = F.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return F.relu(self.fc(similarities))


# 颜色相似网络
class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = (
            nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        )
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (
            torch.arange(0, batch_size * time_window, device=frames.device) << 9
        ).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(
            batch_size * time_window * 512, dtype=torch.int32, device=frames.device
        )
        histograms.scatter_add_(
            0,
            binned_values,
            torch.ones(len(binned_values), dtype=torch.int32, device=frames.device),
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = F.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(
            x, x.transpose(1, 2)
        )  # [batch_size, time_window, time_window]
        similarities_padded = F.pad(
            similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2]
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return F.relu(self.fc(similarities))
        return similarities
