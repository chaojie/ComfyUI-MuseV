from typing import Union, List

import hashlib
import os


def get_md5sum(data: str, length: int = None, blocksize: int=2**23) -> str:
    """获取文件的hash值前几位作为文件唯一标识符

    Args:
        path (str): 文件路径
        length (int, optional): hash前多少位. Defaults to None.
        blocksize (int, optional):分块读取，块的大小. Defaults to None.

    Returns:
        str: hash值前length位
    """
    if isinstance(data, str):
        if os.path.isfile(data):
            signature = get_md5sum_of_file(path=data, length=length, blocksize=blocksize)
        else:
            signature = get_signature_of_string(data, length=length)
    else:
        raise ValueError(
            "only support str or file path str,  but given {}".format(type(data))
        )
    return signature


def get_md5sum_of_file(path: str, length: int = None, blocksize: int=2**23) -> str:
    """获取文件的hash值前几位作为文件唯一标识符

    Args:
        path (str): 文件路径
        length (int, optional): hash前多少位. Defaults to None.
        blocksize (int, optional):分块读取，块的大小. Defaults to None.

    Returns:
        str: hash值前length位
    """

    # sig = (os.popen('md5sum {}'.format(path))).readlines()[0].split('  ')[0]
    m = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(blocksize):
            m.update(chunk)
    sig = m.hexdigest()
    if length is not None:
        sig = sig[:length]
    return sig


def get_signature_of_string(string: str, length: int = None) -> str:
    """cal signature of string

    Args:
        string (str): target string
        length (int, optional): only return the first length character of signature. Defaults to None.

    Returns:
        str: signature of string
    """
    sig = hashlib.md5(string.encode()).hexdigest()
    if length is not None:
        sig = sig[:length]
    return sig
