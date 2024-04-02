from typing import List

import re


def has_key_brace(string: str) -> bool:
    """检测字符串中是否含有{x}。
    注意，不是检测是否有{}

    Args:
        string (str):

    Returns:
        bool:
    """
    flag = re.search("\{.+\}", string)
    flag = flag is not None
    return flag


def merge_near_same_char(string: str, target_char=", ") -> str:
    """合并连续不变的指定字符为1个。如 `1,2,,3,,,4`合并成`1,2,3`

    Args:
        string (str): 待处理的字符串
        target_char (str, optional): 指定的连续字符. Defaults to ",".

    Returns:
        str: 处理后的字符串
    """
    string = re.sub("({}*)+".format(target_char), target_char, string)
    return string


def get_word_from_key_brace_string(string: str, start="{", end="}") -> List:
    """从含有`{key}`的模板字符串中 获取所有的关键词`key`

    Args:
        string (str): 含有`{key}`的模板字符串

    Returns:
        List: 所有关键词 key 列表
    """
    words = re.findall(f"{start}[^{start}|^{end}]+{end}", string)
    words = [word[len(start) : -len(end)] for word in words]
    return words


def clean_str_for_save(string: str, disallowed_chars: List = None):
    if disallowed_chars is None:
        disallowed_chars = r'[\\/:*?"<>|]'
    cleaned_filename = re.sub(disallowed_chars, "", string)
    return cleaned_filename
