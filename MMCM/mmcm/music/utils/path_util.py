import os
from typing import Dict, Tuple

from ...utils.path_util import get_dir_file_map


def get_audio_path_dct(path, exts=["mp3", "flac", "wav"]) -> Dict[str, str]:
    """遍历目标文件夹及子文件夹下所有音频文件，生成字典。"""
    return get_dir_file_map(path, exts=exts)
