from einops import rearrange, repeat
import numpy as np
from copy import deepcopy

import skimage

from ..data.video_dataset import DecordVideoDataset


def hist_match_color(
    image: np.ndarray, reference: np.ndarray, channel_axis: int = -1
) -> np.ndarray:
    """rgb hwc 255

    Args:
        image (np.ndarray): h w c
        reference (np.ndarray): h w c

    Returns:
        np.ndarray: _description_
    """
    res = skimage.exposure.match_histograms(image, reference, channel_axis=channel_axis)
    return res


def hist_match_color_video(
    video: np.ndarray, target: np.ndarray, channel_axis: int = -1
) -> np.ndarray:
    """rgb hw c

    Args:
        video (np.array): t h w c
        target (np.array): h w c

    Returns:
        np.array: t h w c
    """
    new_video = []
    for t in range(len(video)):
        image = hist_match_color(video[t, ...], target, channel_axis=channel_axis)
        new_video.append(image)
    new_video = np.stack(new_video, axis=0)
    return new_video


def hist_match_color_video_batch(
    video: np.ndarray, target: np.ndarray, channel_axis: int = -1
) -> np.ndarray:
    """rgb hw c

    Args:
        video (np.array): b t h w c
        target (np.array): b h w c

    Returns:
        np.array: t h w c
    """
    new_video = []
    for b in range(len(video)):
        image = hist_match_color_video(
            video[b, ...], target[b], channel_axis=channel_axis
        )
        new_video.append(image)
    new_video = np.stack(new_video, axis=0)
    return new_video


def hist_match_color_videodataset(
    video: DecordVideoDataset, target: np.ndarray, channel_axis: int = -1
) -> np.ndarray:
    """rgb t h w c

    Args:
        video (DecordVideoDataset): t h w c
        target (np.ndarray): h w c

    Returns:
        np.ndarray: t h w c
    """
    new_video = []
    for i, batch in enumerate(video):
        batch_data = batch.data
        new_batch = hist_match_color_video(
            batch_data, target, channel_axis=channel_axis
        )
        new_video.append(new_batch)
    new_video = np.concatenate(new_video, axis=0)
    return new_video


def hist_match_video_bcthw(video, target, value: float = 255.0):
    video = rearrange(video, "b c t h w-> b t h w c")
    target = rearrange(target, "b c t h w->(b t) h w c")
    video = (video * value).astype(np.uint8)
    target = (target * value).astype(np.uint8)
    video = hist_match_color_video_batch(video, target)
    video = video / value
    target = target / value
    video = rearrange(video, "b t h w c->b c t h w")
    return video
