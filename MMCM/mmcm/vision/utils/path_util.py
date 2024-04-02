from functools import partial
import os
from typing import Tuple, Union

from ...utils.path_util import get_dir_file_map, get_path_dct
from ...utils.signature import get_md5sum


def get_video_signature(
    path: str,
    rename: bool = False,
    length: int = None,
    signature: str = None,
    sep: str = "@",
) -> Union[str, Tuple[str, str]]:
    """视频文件hash值特殊保存方法，name@signature，若hash值与目标长度不符或者不存在，则重新计算，并根据需要是否修改名字

    Args:
        path (str): 视频源路径
        rename (bool, optional): 是否对视频源文件重命名. Defaults to False.
        length (int, optional): hash值签名长度. Defaults to 8.
        signature (str, optional): full signature of path, to avoid recalculate signature. Defaults to `None`.
        sep (str, optional): 将hash值嵌入命名的分隔符. Defaults to "@".

    Returns:
        str: 视频文件hash值
        Tuple[str, str]: 对应的hash签名, 新的视频文件地址，
    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, ext = os.path.splitext(basename)
    file_signature = None
    if "@" in filename:
        file_signature = filename.split(sep)[-1]
        if length is not None and len(file_signature) != length:
            file_signature = None
    if file_signature is None:
        if signature is None:
            signature = get_md5sum(path, length=length)
            file_signature = signature
        if rename:
            dst_path = os.path.join(
                dirname, "{}@{}{}".format(filename.split(sep)[0], signature, ext)
            )
            os.rename(path, dst_path)
            path = dst_path
    if rename:
        return file_signature, path
    else:
        return file_signature


def split_names_with_signature(basename: str, sep: str = "@") -> str:
    """从videoinfo中分离 video的名字,videoinfo的名字是videoname_hash.json

    Args:
        basename (str): like name_hash
        sep (str): like sep

    Returns:
        str: 视频文件名
    """
    filename = ".".join(basename.split(".")[:-1])
    ext = basename.split(".")[-1]
    if sep in filename:
        filename = sep.join(filename.split(sep)[:-1])
    return filename, ext


def get_video_map_path_dct(
    path: str, mode: int = 1, sep: str = "@", exts=["json"]
) -> dict:
    """遍历目标文件夹及子文件夹下所有视频谱面文件，生成字典。"""
    dct = get_path_dct(
        path=path,
        mode=mode,
        sep=sep,
        split_func=partial(split_names_with_signature, sep=sep),
        exts=exts,
    )
    return dct


def get_video_path_dct(
    path, mode: int = 1, sep: str = "@", exts=["mp4", "mkv", "ts", "rmvb", "mov"]
) -> dict:
    """遍历目标文件夹及子文件夹下所有视频文件，生成字典。"""
    dct = get_path_dct(
        path=path,
        mode=mode,
        sep=sep,
        split_func=partial(split_names_with_signature, sep=sep),
        exts=exts,
    )
    return dct


def get_video_emd_path_dct(path, mode: int = 1, sep: str = "@", exts=["hdf5"]) -> dict:
    """遍历目标文件夹及子文件夹下所有视频文件，生成字典。"""
    dct = get_path_dct(
        path=path,
        mode=mode,
        sep=sep,
        split_func=partial(split_names_with_signature, sep=sep),
        exts=exts,
    )
    return dct
