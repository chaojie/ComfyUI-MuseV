
from collections import namedtuple
from typing import NamedTuple, Tuple, List
import logging
import os
import numpy as np
import subprocess

import requests

import wget 

from .youtube import download_youtube
from .flicker import download_flickr
from .ffmpeg import ffmpeg_load

logger = logging.getLogger(__name__)

# DownloadStatus  = namedtuple("DownloadStatus", ["status_code", "msg"])

status_code = {0: "download: succ",
              -1: "download: failed",
              -2: "clip: failed",
              -3: "directory not exists",
              -4: "skip task",
              - 404: "param error"}


def download_with_request(url, path):
    res = requests.get(url)
    if res.status_code == '200' or res.status_code == 200:
        with open(path, "wb") as f:
            f.write(res.content)
    else:
        print('request failed')
    return path

def download_video(url, save_path:str=None, save_dir:str=None, basename:str=None, filename:str=None, format:str=None, data_type: str="wget", **kwargs) -> Tuple[int, str]:
    if save_path is None:
        if basename is None:
            basename =  f"{filename}.{format}"
        save_path = os.path.join(save_dir, basename)
    if save_dir is None:
        save_dir = os.path.dirname(save_path)
    if basename is None:
        basename = os.path.basename(save_path)
    if filename is None:
        filename, format = os.path.splitext(basename)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path):
        return (-4, save_path)

    try:
        if data_type == "requests":
             save_path = download_with_request(url=url, path=save_path)
        elif data_type == "wget":
            save_path = wget.download(url=url, out=save_path)
        elif data_type == "youtube":
            save_path = download_youtube(url, format=format, save_dir=save_dir, filename=basename)
        elif data_type == "flickr":
            save_path = download_flickr(url, save_path)
        elif data_type == "ffmpeg":
            code = ffmpeg_load(url=url, save_path=save_path)
        else:
            raise ValueError(f"data_type shoulbe one of [wget, youtube, flickr, ffmpeg], but given {data_type}")
    except Exception as e:
        logger.error("failed download file {} to {} failed!".format(url, save_path))
        logger.exception(e)
        return (-1, None)

    return (0, save_path)
