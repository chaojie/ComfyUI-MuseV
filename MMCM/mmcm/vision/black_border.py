# -*- coding: utf-8 -*-

import traceback
import argparse
import os

from moviepy.editor import VideoFileClip
import numpy as np


#   rmBlackBorder: remove the black borders of one image
#   return: cropped image
def det_image_black_border(
    src,  # input image
    thres,  # threshold for cropping: sum([r,g,b] - [0,0,0](black))
    shrink,  # number of pixels to shrink after the blackBorders removed
):
    #   remove the black border on both right and left side
    nRow = src.shape[0]
    nCol = src.shape[1]
    left = 0
    right = nCol

    # for j in range(0, nCol):
    #     if src[:, j].mean() <= thres:
    #         left = j + 1
    #     else:
    #         break
    #
    # for j in range(nCol - 1, -1, -1):
    #     if src[:, j].mean() <= thres:
    #         right = j
    #     else:
    #         break

    black_idx = np.where(src.mean(axis=0) <= thres)[0].tolist()
    for i in black_idx:
        if left < i < nCol // 2:
            left = i
        elif nCol // 2 < i < right:
            right = i

    if right - left > 0:
        left = left + shrink
        right = right - shrink
    else:
        left = 0
        right = nCol

    #   remove the black border on both up and down side
    up = 0
    bottom = nRow

    # for i in range(0, nRow):
    #     if src[i, :].mean() <= thres:
    #         up = i + 1
    #     else:
    #         break
    #
    # for i in range(nRow - 1, -1, -1):
    #     if src[i, :,].mean() <= thres:
    #         bottom = i
    #     else:
    #         break

    black_idx = np.where(src.mean(axis=1) <= thres)[0].tolist()
    for i in black_idx:
        if up < i < nRow // 2:
            up = i
        elif nRow // 2 < i < bottom:
            bottom = i

    if bottom - up > 0:
        top = up + shrink
        bottom = bottom - shrink
    else:
        top = 0
        bottom = nRow

    return (left, top, right, bottom)


def det_video_black_border(video_path):
    video = VideoFileClip(video_path)
    duration = video.duration
    test_duration = 600
    video = video.subclip(
        t_start=duration / 2 - test_duration / 2, t_end=duration / 2 + test_duration / 2
    )
    frame_num = 0
    for frame in video.iter_frames(fps=1):
        frame = frame.astype(np.int64)
        if frame_num == 0:
            frame_sum = frame
        else:
            frame_sum += frame
        frame_num += 1
    frame = frame_sum / frame_num
    return det_image_black_border(frame, 5, 0)
