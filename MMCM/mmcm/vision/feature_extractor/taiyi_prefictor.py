import os
import time
from typing import Union, List, Tuple
from tqdm import tqdm

from PIL import Image
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    CLIPProcessor,
    CLIPModel,
)
import torch
import numpy as np
from numpy import ndarray
from moviepy.editor import VideoFileClip

from ..utils.path_util import get_video_signature
from ..video_map.video_map import VideoMap
from ..data.video_dataset import MoviepyVideoDataset, SequentialDataset
from ...utils.itertools_util import generate_sample_idxs


class ClipVisionFeatureExtractor(object):
    def __init__(self, model_name: str, local_file: bool = True, device: str = "cpu"):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = (
            CLIPModel.from_pretrained(model_name, local_files_only=local_file)
            .eval()
            .to(self.device)
        )
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=local_file
        )

    def image(self, img_paths):
        image = self.processor(
            images=[Image.open(i) for i in img_paths], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy()

    def __call__(self, image):
        return self.image(image)

    def predict_images(
        self, image: Union[Image.Image, List[Image.Image], torch.Tensor]
    ) -> np.ndarray:
        if isinstance(image, str):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], str):
            image = [Image.open(i) for i in image]
        image = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy()

    def predict_clip(
        self, clip: Union[Image.Image, List[Image.Image], torch.Tensor], batch_size: int
    ) -> ndarray:
        features = []
        num = len(clip)
        windows = generate_sample_idxs(
            num, window_size=batch_size, step=batch_size, drop_last=False
        )
        for i, window in enumerate(windows):
            sub_clip = clip[window]
            feature = self.predict_images(sub_clip)
            features.append(feature)
        feature = np.concatenate(features, axis=0)
        return features

    def predict_video(
        video: Union[str, SequentialDataset],
        video_map: VideoMap,
        vf_extractor,
        bbx_extr,
        time_size: int = None,
        step: int = None,
        overlap: int = None,
        sample_rate: int = None,
        drop_last: bool = False,
        max_frame_num_per_clip: int = 5,
    ):
        # prepare video
        if isinstance(video, str):
            video = MoviepyVideoDataset(
                video,
                time_size=time_size,
                step=step,
                overlap=overlap,
                drop_last=drop_last,
                sample_rate=sample_rate,
            )
        if video_map.meta_info.content_box != video.content_box:
            video.content_box = video_map.content_box
        fps = 1
        max_frame_num = 5
        select_frame_idx = []
        select_frame_clip = []
        for i in range(len(video_map.clipseq)):
            clip = video_map.clipseq[i]
            if clip["cliptype"] == "transition":
                continue
            select_frame_num = int(min(np.ceil(clip["duration"] * fps), max_frame_num))
            clip_total_frame_num = clip["frame_end"] - clip["frame_start"]
            frame_duration = clip_total_frame_num // (select_frame_num + 1)
            for j in range(select_frame_num):
                select_frame_idx.append(clip["frame_start"] + (j + 1) * frame_duration)
                select_frame_clip.append(i)

        return video_map


class TaiyiVisionFeatureExtractor(ClipVisionFeatureExtractor):
    def __init__(
        self,
        model_name: str = "clip-vit-large-patch14",
        local_file: bool = True,
        device: str = "cpu",
    ):
        """_summary_

        Args:
            model_name (str, optional): clip-vit-large-patch14 or openai/clip-vit-large-patch14. Defaults to "clip-vit-large-patch14".
            local_file (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        super().__init__(model_name, local_file, device)
