from __future__ import print_function

import traceback
from typing import Dict
from moviepy.editor import VideoFileClip
import hashlib
import json
import numpy as np
import os
import time
import copy
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import librosa

from ...utils.util import load_dct_from_file

# from lgss.utilis.package import *

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformer = transforms.Compose(
    [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ]
)


def wav2stft(data):
    # normalize
    mean = (data.max() + data.min()) / 2
    span = (data.max() - data.min()) / 2
    if span < 1e-6:
        span = 1
    data = (data - mean) / span  # range: [-1,1]

    D = librosa.core.stft(data, n_fft=512)
    freq = np.abs(D)
    freq = librosa.core.amplitude_to_db(freq)
    span = 80
    thr = 4 * span

    if freq.shape[1] <= thr:
        copy_ = freq.copy()
        while freq.shape[1] < thr:
            tmp = copy_.copy()
            freq = np.concatenate((freq, tmp), axis=1)
        freq = freq[:, :thr]
    else:
        # sample
        n = freq.shape[1]
        stft_img = []
        stft_img.append(freq[:, : 2 * span])
        # stft_img.append(freq[:, n//2 - span : n//2 + span])
        stft_img.append(freq[:, -2 * span :])
        freq = np.concatenate(stft_img, axis=1)
    return freq


def test(
    model,
    data_place,
    data_cast=None,
    data_act=None,
    data_aud=None,
    last_image_overlap_feat=None,
    last_aud_overlap_feat=None,
):
    with torch.no_grad():
        # data_place = data_place.cuda()  if data_place is not None else []
        data_cast = data_cast.cuda() if data_cast is not None else []
        data_act = data_act.cuda() if data_act is not None else []
        data_aud = data_aud.cuda() if data_aud is not None else []
        (
            img_output,
            aud_output,
            image_overlap_feat,
            audio_overlap_feat,
            shot_dynamic_list,
        ) = model(
            data_place,
            data_cast,
            data_act,
            data_aud,
            last_image_overlap_feat,
            last_aud_overlap_feat,
        )
        img_output = img_output.view(-1, 2)
        img_output = F.softmax(img_output, dim=1)
        img_prob = img_output[:, 1]
        img_prob = img_prob.cpu()
        aud_output = aud_output.view(-1, 2)
        aud_output = F.softmax(aud_output, dim=1)
        aud_prob = aud_output[:, 1]
        aud_prob = aud_prob.cpu()
    return img_prob, aud_prob, image_overlap_feat, audio_overlap_feat, shot_dynamic_list


def predict(
    model,
    cfg,
    video_path,
    save_path,
    map_path,
    seq_len=120,
    shot_num=4,
    overlap=21,
    shot_frame_max_num=60,
):
    assert overlap % 2 == 1
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
    if os.path.exists(save_path) and not args.overwrite:
        video_map = json.load(open(save_path), encoding="UTF-8")
        valid_clips = []
        for clip in video_map["clips"]:
            if clip["cliptype"] == "body" and clip["duration"] > 0.25:
                valid_clips.append(clip)
        # Capture video
        if (
            video_map["content_box"][2] - video_map["content_box"][0]
            > video_map["content_box"][3] - video_map["content_box"][1]
        ):
            target_resolution = (
                256
                * video_map["height"]
                / (video_map["content_box"][3] - video_map["content_box"][1]),
                None,
            )
        else:
            target_resolution = (
                None,
                256
                * video_map["width"]
                / (video_map["content_box"][2] - video_map["content_box"][0]),
            )
        video = VideoFileClip(
            video_path,
            target_resolution=target_resolution,
            resize_algorithm="bilinear",
            audio_fps=16000,
        )
        # video = video.crop(*video_map["content_box"])
        x1 = video_map["content_box"][0] * video.size[0] // video_map["width"]
        y1 = video_map["content_box"][1] * video.size[1] // video_map["height"]
        x2 = video_map["content_box"][2] * video.size[0] // video_map["width"]
        y2 = video_map["content_box"][3] * video.size[1] // video_map["height"]
        video = video.crop(
            width=(x2 - x1) * 224 / 256,
            height=224,
            x_center=(x1 + x2) // 2,
            y_center=(y1 + y2) // 2,
        )
        print("exists " + save_path)
    else:
        map_path = os.path.join(
            map_path, "{}_{}.json".format(video_name, video_hash_code[:8])
        )
        if not os.path.exists(map_path):
            print("map not exist: ", map_path)
            return

        video_map = json.load(open(map_path), encoding="UTF-8")
        assert video_hash_code == video_map["video_file_hash_code"]

        # Capture video
        if (
            video_map["content_box"][2] - video_map["content_box"][0]
            > video_map["content_box"][3] - video_map["content_box"][1]
        ):
            target_resolution = (
                256
                * video_map["height"]
                / (video_map["content_box"][3] - video_map["content_box"][1]),
                None,
            )
        else:
            target_resolution = (
                None,
                256
                * video_map["width"]
                / (video_map["content_box"][2] - video_map["content_box"][0]),
            )
        video = VideoFileClip(
            video_path,
            target_resolution=target_resolution,
            resize_algorithm="bilinear",
            audio_fps=16000,
        )
        # video = video.crop(*video_map["content_box"])
        x1 = video_map["content_box"][0] * video.size[0] // video_map["width"]
        y1 = video_map["content_box"][1] * video.size[1] // video_map["height"]
        x2 = video_map["content_box"][2] * video.size[0] // video_map["width"]
        y2 = video_map["content_box"][3] * video.size[1] // video_map["height"]
        video = video.crop(
            width=(x2 - x1) * 224 / 256,
            height=224,
            x_center=(x1 + x2) // 2,
            y_center=(y1 + y2) // 2,
        )
        fps = video.fps
        duration = video.duration
        total_frames = int(duration * fps)
        width, height = video.size
        print("fps, frame_count, width, height:", fps, total_frames, width, height)

        valid_clips = []
        for clip in video_map["clips"]:
            if clip["cliptype"] == "body" and clip["duration"] > 0.25:
                valid_clips.append(clip)
        # valid_clips = valid_clips[:150]
        total_shot_num = len(valid_clips)
        last_image_overlap_feat = None
        last_aud_overlap_feat = None
        truncate_time = 0.1
        all_shot_dynamic_list = []
        for i in range(total_shot_num // (seq_len - overlap) + 1):
            shot_frame_list = []
            shot_audio_list = []
            start_shot = i * (seq_len - overlap)
            end_shot = min(start_shot + seq_len, total_shot_num)
            if i != 0:
                start_shot += overlap
            print(start_shot, end_shot)
            if start_shot >= end_shot:
                break
            for clip in valid_clips[start_shot:end_shot]:
                time_start = clip["time_start"]
                time_end = clip["time_start"] + clip["duration"]
                truncate_time = min(clip["duration"] / 10, 0.1)
                time_start += truncate_time
                time_end -= truncate_time
                time_start = max(time_start, (time_end + time_start) / 2 - 3)
                time_end = min(time_end, (time_end + time_start) / 2 + 3)
                duration = time_end - time_start
                t0 = time.time()
                video_subclip = video.subclip(time_start, time_end)
                # video_save_path = os.path.join(args.video_save_path, 'shot_{:04d}.mp4'.format(clip["clipid"]))
                # video_subclip.write_videofile(video_save_path, threads=8, codec='libx264')
                if "image" in cfg.dataset["mode"]:
                    frame_iter = video_subclip.iter_frames(fps=10)
                    shot_frame = []
                    for frame in frame_iter:
                        frame = transformer(frame)
                        shot_frame.append(frame)
                        if len(shot_frame) > shot_frame_max_num:
                            break
                    shot_frame = torch.stack(shot_frame)
                    shot_frame = shot_frame.cuda()
                    shot_frame_list.append(shot_frame)

                t5 = time.time()
                if "aud" in cfg.dataset["mode"]:
                    try:
                        sub_audio = video.audio.subclip(
                            clip["time_start"], clip["time_start"] + clip["duration"]
                        )
                        sub_audio = sub_audio.to_soundarray(
                            fps=16000, quantize=True, buffersize=20000
                        )
                        sub_audio = sub_audio.mean(axis=1)
                    except:
                        sub_audio = np.zeros((16000 * 4), np.float32)
                    sub_audio = wav2stft(sub_audio)
                    sub_audio = torch.from_numpy(sub_audio).float()
                    sub_audio = sub_audio.unsqueeze(dim=0)
                    shot_audio_list.append(sub_audio)
                t6 = time.time()
                print(clip["clipid"], t5 - t0, t6 - t5)

            data_place = data_aud = None
            if len(shot_frame_list) > 0:
                # data_place = torch.stack(shot_frame_list)
                data_place = shot_frame_list
            if len(shot_audio_list) > 0:
                data_aud = torch.stack(shot_audio_list)
                data_aud = data_aud.unsqueeze(dim=0)
            (
                img_preds,
                aud_preds,
                last_image_overlap_feat,
                last_aud_overlap_feat,
                shot_dynamic_list,
            ) = test(
                model,
                data_place=data_place,
                data_aud=data_aud,
                last_image_overlap_feat=last_image_overlap_feat,
                last_aud_overlap_feat=last_aud_overlap_feat,
            )
            print(shot_dynamic_list)
            all_shot_dynamic_list.extend(shot_dynamic_list)
            if total_shot_num > end_shot:
                if i == 0:
                    img_preds_all = img_preds[: -(overlap - shot_num + 1) // 2]
                    aud_preds_all = aud_preds[: -(overlap - shot_num + 1) // 2]
                else:
                    img_preds_all = torch.cat(
                        (
                            img_preds_all,
                            img_preds[
                                (overlap - shot_num + 1)
                                // 2 : -(overlap - shot_num + 1)
                                // 2
                            ],
                        ),
                        dim=0,
                    )
                    aud_preds_all = torch.cat(
                        (
                            aud_preds_all,
                            aud_preds[
                                (overlap - shot_num + 1)
                                // 2 : -(overlap - shot_num + 1)
                                // 2
                            ],
                        ),
                        dim=0,
                    )
            else:
                if i == 0:
                    img_preds_all = img_preds
                    aud_preds_all = aud_preds
                else:
                    img_preds_all = torch.cat(
                        (img_preds_all, img_preds[(overlap - shot_num + 1) // 2 :]),
                        dim=0,
                    )
                    aud_preds_all = torch.cat(
                        (aud_preds_all, aud_preds[(overlap - shot_num + 1) // 2 :]),
                        dim=0,
                    )

        print(
            img_preds_all.shape[0],
            total_shot_num - shot_num + 1,
            len(all_shot_dynamic_list),
        )
        assert img_preds_all.shape[0] == total_shot_num - shot_num + 1
        assert len(all_shot_dynamic_list) == total_shot_num
        print("img_preds_all: ", img_preds_all)
        print("aud_preds_all: ", aud_preds_all)
        video_map["scenes_img_preds"] = img_preds_all.tolist()
        video_map["scenes_aud_preds"] = aud_preds_all.tolist()
        for clip, dynamic in zip(valid_clips, all_shot_dynamic_list):
            clip["dynamic"] = None
            if dynamic is not None:
                clip["dynamic"] = round(np.clip(dynamic, 0, 1), 5)

    preds_all = cfg.model.ratio[0] * np.array(
        video_map["scenes_img_preds"]
    ) + cfg.model.ratio[3] * np.array(video_map["scenes_aud_preds"])
    video_map["scenes_preds"] = preds_all.tolist()
    scene_boundary = np.where(preds_all > args.threshold)[0]
    video_map["scenes"] = []
    scene = {
        "sceneid": 0,
        "clip_start": valid_clips[0]["clipid"],
        "clip_end": valid_clips[0]["clipid"],
        "time_start": valid_clips[0]["time_start"],
        "time_end": valid_clips[0]["time_start"] + valid_clips[0]["duration"],
    }
    for i in scene_boundary:
        scene["clip_end"] = valid_clips[i + shot_num // 2 - 1]["clipid"]
        scene["time_end"] = (
            valid_clips[i + shot_num // 2 - 1]["time_start"]
            + valid_clips[i + shot_num // 2 - 1]["duration"]
        )
        scene["roles"] = {}
        scene["dynamic"] = None
        dynamic_num = 0
        dynamic = 0
        for clip in video_map["clips"][scene["clip_start"] : scene["clip_end"] + 1]:
            for roleid in clip["roles"].keys():
                if roleid not in scene["roles"]:
                    scene["roles"][roleid] = {
                        "name": clip["roles"][roleid]["name"]
                        if "name" in clip["roles"][roleid]
                        else ""
                    }
            if "dynamic" in clip and clip["dynamic"] != None:
                dynamic += clip["dynamic"]
                dynamic_num += 1
        if dynamic_num > 0:
            scene["dynamic"] = dynamic / dynamic_num
        for clip in video_map["clips"][scene["clip_start"] : scene["clip_end"] + 1]:
            clip["scene_roles"] = scene["roles"]
            clip["scene_dynamic"] = scene["dynamic"]
            clip["sceneid"] = scene["sceneid"]
        video_map["scenes"].append(copy.deepcopy(scene))
        scene["sceneid"] += 1
        scene["clip_start"] = scene["clip_end"] = valid_clips[i + shot_num // 2][
            "clipid"
        ]
        scene["time_start"] = valid_clips[i + shot_num // 2]["time_start"]
        scene["time_end"] = (
            valid_clips[i + shot_num // 2]["time_start"]
            + valid_clips[i + shot_num // 2]["duration"]
        )
    scene["clip_end"] = valid_clips[-1]["clipid"]
    scene["time_end"] = valid_clips[-1]["time_start"] + valid_clips[-1]["duration"]
    scene["roles"] = {}
    scene["dynamic"] = None
    dynamic_num = 0
    dynamic = 0
    for clip in video_map["clips"][scene["clip_start"] : scene["clip_end"] + 1]:
        for roleid in clip["roles"].keys():
            if roleid not in scene["roles"]:
                scene["roles"][roleid] = {
                    "name": clip["roles"][roleid]["name"]
                    if "name" in clip["roles"][roleid]
                    else ""
                }
            if "dynamic" in clip and clip["dynamic"] != None:
                dynamic += clip["dynamic"]
                dynamic_num += 1
        if dynamic_num > 0:
            scene["dynamic"] = dynamic / dynamic_num
    for clip in video_map["clips"][scene["clip_start"] : scene["clip_end"] + 1]:
        clip["scene_roles"] = scene["roles"]
        clip["scene_dynamic"] = scene["dynamic"]
        clip["sceneid"] = scene["sceneid"]
    video_map["scenes"].append(scene)
    return video_map


class SceneTransitionPredictor(object):
    def __init__(self, config_path, overlap=41, model_path=None) -> None:
        from mmcv import Config
        from lgss.utilis import load_checkpoint
        import lgss.src.models as models

        self.config_path = config_path
        cfg = Config.fromfile(config_path)
        # cfg = load_dct_from_file(config_path)
        self.cfg = cfg
        self.model = models.__dict__[cfg.model.name](cfg, overlap).cuda()
        self.model = nn.DataParallel(self.model)
        checkpoint = load_checkpoint(
            osp.join(cfg.logger.logs_dir, "model_best.pth.tar")
        )
        paras = {}
        for key, value in checkpoint["state_dict"].items():
            if key in self.model.state_dict():
                paras[key] = value
        if "aud" in cfg.dataset["mode"]:
            c_logs_dir = cfg.logger.logs_dir.replace("image50", "aud")
            checkpoint = load_checkpoint(osp.join(c_logs_dir, "model_best.pth.tar"))
            for key, value in checkpoint["state_dict"].items():
                if key in self.model.state_dict():
                    paras[key] = value
        print(list(paras.keys()))
        self.model.load_state_dict(paras)
        self.model.eval()

    def __call__(
        self,
        video_path,
        video_map,
    ) -> Dict:
        video_info = predict(
            self.model,
            self.cfg,
            video_path,
            video_map,
            overlap=self.overlap,
        )
        return video_info
