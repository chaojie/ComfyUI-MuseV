import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
from moviepy.editor import VideoFileClip


# TODO: 待后续设计、处理
class OpenCVVideoDataset(Dataset):
    def __init__(
        self,
        video_path,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

    def __call__(self, idx) -> np.array:
        self.cap.set(2, idx)
        frame = self.cap.read()
        return frame

    def close(self):
        self.cap.release()


class MoviepyVideoDataset(Dataset):
    def __init__(self, video_path, mode="time") -> None:
        self.video_path = video_path
        self.videoclip = VideoFileClip(video_path)
        self.mode = mode

    def __call__(self, t):
        if self.mode == "int":
            t = t / self.videoclip.fps
        frame = self.videoclip.get_frame(t)
        return frame

    def __len__(self):
        n_total = self.videoclip.duration * self.videoclip.fps
        return n_total


def generate_videoclip_batchsampler(video_info):
    sampler = []
    fps = video_info["fps"]
    for i, clip in enumerate(video_info["slices"]):
        time_start = clip["time_start"]
        duration = clip["duration"]
        n_start = int(time_start * fps)
        n_frame = int(duration * fps)
        sampler.append(range(n_start, n_frame, 1))
    return sampler


class VideoClipBatchSampler(Sampler):
    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def __iter__(self):
        return iter(self.sampler)


def iter_videoclip(model, videoinfo):
    pass


if __name__ == "__main__":
    import json

    PROJECT_DIR = os.path.join(os.path.dirname(__file__), "../..")
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    TEST_IMG_PATH = os.path.join(DATA_DIR, "KDA_ALLOUT.jpeg")
    TEST_VIDEO_PATH = os.path.join(DATA_DIR, "video.mp4")
    TEST_VIDEOMAP_PATH = os.path.join(DATA_DIR, "videomap_大鱼海棠.json")
    with open(TEST_VIDEOMAP_PATH, "r") as f:
        videoinfo = json.load(f)
    videoinfo["fps"] = 30
    sampler = generate_videoclip_batchsampler(videoinfo)
    videoclip_batchsampler = VideoClipBatchSampler(sampler=sampler)
    print("videoclip_batchsampler length", videoclip_batchsampler)
    for i, batch in enumerate(videoclip_batchsampler):
        print(i, batch)
