# -*- coding: utf-8 -*-
"""
Created on Tue Sep 8 13:31:25 2020

@author: infguo
"""

import os
from typing import Dict
import argparse
import json
import traceback
import hashlib

import cv2
from moviepy.editor import VideoFileClip

# TODO
# 该部分与源代码相比，修改了insight_face的输出接口，将性别分数透传出来，用于后续更精准的决策


def inference(frame, app, max_face=10):
    # Start to perform face recognition
    try:  # Handle exception
        faces = app.get(frame, max_num=max_face)
    except Exception as e:
        print("is discarded due to exception {}!".format(e))
        return

    if (
        len(faces) == 0
    ):  # If the landmarks cannot be detected, the img will be discarded
        return
    return faces


def predict(app, video_path, video_map, sample_fps):
    from insightface.app import FaceAnalysis

    video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    # video_hash_code = (os.popen('md5sum {}'.format(video_path))).readlines()[0].split('  ')[0]
    with open(video_path, "rb") as fd:
        data = fd.read()
    video_hash_code = hashlib.md5(data).hexdigest()

    assert video_hash_code == video_map["video_file_hash_code"]

    # Capture video
    video = VideoFileClip(video_path)
    video = video.crop(*video_map["content_box"])
    fps = video.fps
    duration = video.duration
    total_frames = int(duration * fps)
    width, height = video.size
    print("fps, frame_count, width, height:", fps, total_frames, width, height)

    video_map["detect_fps"] = sample_fps
    video_map["face_detections"] = []

    cnt_frame, step = 0, 0

    fps = video_map["sample_fps"]
    for frame in video.iter_frames(fps=fps):
        if cnt_frame >= step:
            step += fps / sample_fps
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = inference(frame, app, max_face=0)
            if faces and len(faces) > 0:
                for f in faces:
                    f["bbox"] = f["bbox"].tolist()
                    f["kps"] = f["kps"].tolist()
                    f["embedding"] = f["embedding"].tolist()
                    f["det_score"] = str(f["det_score"])
                    f["gender"] = str(f["gender"])
                    f["age"] = str(f["age"])
            else:
                faces = None
            video_map["face_detections"].append(
                {"frame_idx": cnt_frame, "faces": faces}
            )

        cnt_frame += 1
    return video_map


class InsightfacePredictor(object):
    def __init__(
        self,
        sample_fps=10,
    ) -> None:
        # Load models
        self.sample_fps = sample_fps
        self.app = FaceAnalysis(
            allowed_modules=["detection", "genderage", "recognition"],
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": "0"}],
        )
        self.app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))

    def __call__(self, video_path, video_map) -> Dict:
        video_info = predict(video_path, video_map, sample_fps=self.sample_fps)
        return video_info
