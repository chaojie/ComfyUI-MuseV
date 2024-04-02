# -*- coding: UTF-8 -*-

"""
__author__ = zhiqiangxia
__date__ = 2019-03-18
"""

import os
import time
import json
import importlib.util
from typing import Any, Tuple, Dict, List, Iterable
from collections import Counter
import pandas as pd

import yaml
import pandas as pd


def dict2list(dct: dict) -> list:
    """将字典转换为列表，若值为列表，使用extend而不是append

    Args:
        dct (dict):

    Returns:
        list:
    """
    lst = []
    for k, v in dct.items():
        if isinstance(v, list):
            lst.extend(v)
        else:
            lst.append(v)
    return lst


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self) -> float:
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average: bool = True) -> float:
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def load_dct_from_file(path: str, key=None) -> dict:
    """读取字典类型的文件

    Args:
        path (str): 字典文件路径

    Raises:
        ValueError: 不支持该字典文件类型，仅支持json、yaml、python中的字典key

    Returns:
        dict: 读取的字典
    """
    if path.endswith(".json"):
        dct = load_json(path)
    elif path.endswith(".yaml"):
        dct = load_yaml(path)
    elif path.endswith(".py"):
        dct = load_edct_py(path, key)
    else:
        raise ValueError("unsupported config file")
    return dct


def load_json(path: str) -> dict:
    """读取json文件

    Args:
        path (str): json路径

    Returns:
        dict: 读取后的python 字典
    """
    with open(path, "r", encoding="utf-8") as f:
        dct = json.load(f)
    return dct


def load_yaml(path: str) -> dict:
    """读取yaml文件

    Args:
        path (str): yaml路径

    Returns:
        dict: 读取后的python 字典
    """
    dct = yaml.load(path)
    return dct


def load_edct_py(path: str, obj_name: str = None) -> dict:
    """读取Python中的字典

    Args:
        path (str): py文件路径
        obj_name (str): py文件中的字典变量名

    Returns:
        dict: 读取后的字典
    """
    module_name = "module_name"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dct = module if obj_name is None else getattr(module, obj_name)
    return dct


def merge_dct(target_dct: dict, source_dct: dict = None) -> None:
    """
    merge source_dct into target_dct
    """
    if source_dct is not None:
        for k, v in source_dct.items():
            if k not in target_dct:
                target_dct[k] = v
            else:
                if not isinstance(v, dict):
                    target_dct[k] = v
                else:
                    merge_dct(target_dct[k], source_dct[k])


def convert_class_attr_to_dict(
    obj: object, target_keys: list = None, ignored_keys: list = None
) -> dict:
    """将类中的属性转化成字典，默认转化为所有属性。

    Args:
        obj (object): 类对象
        target_keys (list, optional): 需要保存的属性. Defaults to None.
        ignored_keys (list, optional): 需要忽视的属性. Defaults to None.

    Returns:
        dict: 转换后的字典
    """
    if target_keys is not None:
        dct = {k: v for k, v in obj.__dict__.items() if k in target_keys}
        return dct
    if ignored_keys is None:
        ignored_keys = []
    dct = {k: v for k, v in obj.__dict__.items() if k not in ignored_keys}
    return dct


def merge_list_continuous_same_element(lst: List[Any]) -> List[Dict[str, Any]]:
    """将一层列表的相邻值合并，并返回每一个不同值的stat、end、元素值

    Args:
        lst (List[Any]): _description_

    Returns:
        List[Dict[str, Any]]: 合并后的列表结果，形如
        [
            {
                "star": x,
                "end": x,
                "element": x,
            },
        ]
    """
    merge_lst = []
    if len(lst) == 0:
        return lst
    elif len(lst) == 1:
        return {"start": 0, "end": 0, "element": lst[0]}
    start = 0
    end = 0
    last_element = lst[end]
    for i, element in enumerate(lst):
        if i == 0:
            continue
        if i == len(lst) - 1:
            if element != last_element:
                dct = {"start": start, "end": end, "element": last_element}
                merge_lst.append(dct)
                last = {"start": len(lst) - 1, "end": i, "element": element}
                merge_lst.append(last)
            else:
                last = {"start": start, "end": i, "element": element}
                merge_lst.append(last)
            break

        if element != last_element:
            dct = {"start": start, "end": end, "element": last_element}
            merge_lst.append(dct)
            start = i
            last_element = element
        end = i
    return merge_lst


def flatten2generator(lst: Iterable, ignored_iterable_types: List = None):
    """将一个嵌套迭代器展开成生成器，

    Args:
        lst (Iterable): 待展开的迭代器
        ignored_iterable_types (_type_, List): 如果待展开的迭代器在该目标列表中，则不展开. Defaults to None.

    Yields:
        _type_: 不是迭代器的类型，或者 ignored_iterable_types中的类型
    """
    if ignored_iterable_types is None:
        ignored_iterable_types = []
    for element in lst:
        if (
            isinstance(element, Iterable)
            and type(element) not in ignored_iterable_types
        ):
            for subc in flatten2generator(element):
                yield subc
        else:
            yield element


def flatten(lst: List, ignored_iterable_types=None) -> List:
    """将 flatten_nested_iterable_2_generator展开的生成器转化为迭代器，容器目前使用 list

    Args:
        lst (List): _description_
        ignored_iterable_types (_type_, List): 如果待展开的迭代器在该目标列表中，则不展开. Defaults to None.

    Returns:
        List: _description_
    """
    return list(flatten2generator(lst, ignored_iterable_types=ignored_iterable_types))


def get_current_strtime(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    current_time = time.strftime(fmt, time.localtime())
    return current_time


def advanced_count(df: Iterable) -> Dict:
    """对迭代器中的数值内容进行统计

    Args:
        df (Iterable): 值为可统计的迭代器，如str, int, float等

    Returns:
        Dict: 统计结果
    """
    n_all = len(df)
    count = Counter(df)
    new_count = {"total": n_all}
    for k, v in count.items():
        new_count[k] = v
        new_count["{}_ratio".format(k)] = round(v / n_all * 100, 2)
    return new_count


class CustomCounter(object):
    def __init__(self, name: str) -> None:
        """多类别统计器，支持输入值的类别，针对每种类别分别统计

        Args:
            name (str): _description_
        """
        self.name = name
        self._category_col = "category"
        self._value_col = "value"
        self._df = pd.DataFrame(columns=[self._category_col, self._value_col])

    def update(self, v, k: str = "default") -> None:
        new = pd.DataFrame([{self._category_col: k, self._value_col: v}])
        self._df = pd.concat([self._df, new], axis=0)

    def advanced_count(
        self,
    ) -> Dict:
        dct = {"total": self.simple_count()}
        if len(self._df[self._category_col] != "default") > 0:
            for k, k_df in self._df.groupby(self._category_col):
                dct[k] = advanced_count(k_df[self._value_col])
        return dct

    def simple_count(
        self,
    ) -> Dict:
        return advanced_count(self._df[self._value_col])

    def count(self, is_simple: bool = False) -> Dict:
        if is_simple:
            return self.simple_count()
        else:
            return self.advanced_count()
