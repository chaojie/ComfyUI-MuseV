import os

from typing import Callable


def download_data(src: str, dst: str = None, download_func: Callable = None) -> str:
    """使用download_func将目标文件下载到目标路径下

    Args:
        src (str): _description_
        dst (str, optional): _description_. Defaults to None.
        download_func (Callable, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    if not os.path.exists(dst):
        download_func(src, dst)
    return dst


def download_data_with_cos(src: str, dst: str) -> None:
    """使用cos工具下载cos上的文件

    Args:
        src (str): 原目录，
        dst (str): 目标目录，暂不支持修改后的目录名字
    """
    from cos_utils.crate import CosCrate

    src_basename = os.path.basename(src)
    dst_path = os.path.join(dst, src_basename)
    if os.path.exists(dst_path):
        print("existed: {}".format(dst_path))
        return
    if "." not in src_basename:
        if src[-1] != "/":
            src += "/"
    if "." not in os.path.basename(dst):
        if dst[-1] != "/":
            dst += "/"
    CosCrate().download_to_local(src, dst)
    