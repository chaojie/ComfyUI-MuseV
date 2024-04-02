from typing import List, Dict, Sequence, Union, Tuple, Any, Hashable

import numpy as np


def pick_subdct(src: Dict[Hashable, Any], target_keys: List[str] = None, ignored_keys: List[str] = None) -> Dict[Hashable, Any]:
    """提取字典中的目标子字典

    Args:
        src (Dict[Hashable, Any]): 原字典
        target_keys (List[str], optional): 目标key. Defaults to None.
        ignored_keys (List[str], optional): 忽略的key. Defaults to None.

    Returns:
        Dict[Hashable, Any]: 子字典
    """
    dst = {}
    if target_keys is not None:
        for k in target_keys:
            if k in src:
                dst[k] = src[k]
    if ignored_keys is not None:
        for k in src:
            if k not in ignored_keys:
                dst[k] = src[k]
    return dst


def str2intlist(string: str, sep: str="_", range_sep: str=":", discrete_sep=",") -> List[int]:
    """将1:2_3:4_5,6,7的字符串转化成整数索引列表，方便取子任务, 左闭右比

    Args:
        string (str): 输入字符串

    Returns:
        List: 转化后的整数列表
    """
    string = string.split(sep)
    lst = []
    for s in string:
        if range_sep in s:
            # 采用左闭、右闭方式
            start, end = [int(x) for x in s.split(range_sep)]
            sub_lst = range(start, end + 1)
        else:
            sub_lst = [int(x) for x in s.split(discrete_sep)]
        lst.extend(sub_lst)
    lst = sorted(set(lst))
    return lst


def dict_has_keys(dct: Dict[Hashable, Any], keys: List[Union[str, int]]) -> bool:
    """嵌套字典是否有嵌套key

    Args:
        dct (Dict[Hashable, Any]): 有多层嵌套的字典
        keys (List[Union[str, int]]): 字符串列表，从前往后表示嵌套字典key

    Returns:
        bool: dct是否有keys
    """
    if keys[0] not in dct:
        # if not hasattr(dct, keys[0]):
        return False
    else:
        if len(keys) == 1:
            return True
        else:
            return dict_has_keys(dct[keys[0]], keys[1:])


def dict_get_keys(dct: Dict[Hashable, Any], keys: List[Union[str, int]]) -> Any:
    """根据索引列表获取嵌套字典的值

    Args:
        dct (Dict[Hashable, Any]): 嵌套字典
        keys (List[Union[str, int]]): 用列表表示的嵌套索引

    Returns:
        Any: 嵌套索引keys对应的值
    """
    if len(keys) == 1:
        return dct[keys[0]]
    else:
        return dict_get_keys(dct[keys[0]], keys[1:])
