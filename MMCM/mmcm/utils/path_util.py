import os
from typing import Tuple, Callable, Dict
from functools import partial


def get_file_name_ext(basename: str) -> Tuple[str, str]:
    """分离文件名和后缀，适用于复杂命名的分离，目前支持
    含.的名字

    Args:
        basename (str): xx.xxx.xx.ext

    Returns:
        Tuple[str, str]: name, ext
    """
    ext = basename.split(".")[-1]
    name = ".".join(basename.split(".")[:-1])
    return name, ext


def get_dir_file_map(
    path, split_func=None, filter_func: Callable = None, exts: list = None
) -> dict:
    """遍历目标文件夹及子文件夹下所有符合后缀目标的文件，生成字典。
    split_func 可以用于对文件名做处理生成想要的字典key。

    Args:
        path (str): 目标文件夹
        split_func (__call__, optional): 可以用于对文件名做处理生成想要的字典key。. Defaults to None.
        exts (list, optional): 目标文件后缀, 例如["mp3", "json"]. Defaults to None.

    Returns:
        dict: key是处理后的文件名，值是绝对路径
    """
    dct = {}
    for rootdir, dirnames, basenames in os.walk(path):
        for basename in basenames:
            path = os.path.join(rootdir, basename)
            if filter_func is not None:
                if not filter_func(path):
                    continue
            if split_func is None:
                ext = basename.split(".")[-1]
                filename = ".".join(basename.split(".")[:-1])
            else:
                filename, ext = split_func(basename)
            if exts is None:
                dct[filename] = path
            else:
                if ext.lower() in exts:
                    dct[filename] = path
    return dct


def get_path_dct(path, exts, mode: int=1, split_func: Callable=None, sep: str="@") -> Dict[str, str]:
    """遍历目标文件夹及子文件夹下所有视频文件，生成字典。"""
    if mode == 1:
        dct = get_dir_file_map(path, exts=exts)
    elif mode == 2:
        dct = get_dir_file_map(path, split_func=split_func, exts=exts)
    elif mode== 3:
        dct = get_path_dct(path, mode=1, sep=sep, exts=exts)
        dct2 = get_path_dct(path, mode=2, sep=sep, split_func=split_func, exts=exts)
        dct.update(**dct2)
    else:
        raise ValueError("only support mode 1, 2, 3")
    return dct


