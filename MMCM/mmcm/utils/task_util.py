from copy import deepcopy
from itertools import product
import os
from typing import Dict, List
import logging

import pandas as pd

from .path_util import get_dir_file_map
from .signature import get_signature_of_string

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def generate_tasks(
    path: str,
    key: str = None,
    sep: str = ",",
    exts: List[str] = None,
    subset_row: str = None,
) -> List[Dict]:
    """读取文件，生成任务表格

    Args:
        path (str): 任务文件路径
        key (str, optional): 作为任务名的字段. Defaults to None.
        sep (str, optional): 表格字段分隔符. Defaults to ",".
        exts (List[str], optional): 如果是文件夹，目前文件类型. Defaults to None.
        subset_row (str, optional): 将1:2_3:4的字符串转化成整数索引列表，方便取子任务. Defaults to None.

    Returns:
        List[Dict]: 列表后的任务字典列表
    """
    if os.path.isdir(path):
        tasks = get_dir_file_map(path=path, exts=exts)
        tasks = [{key: k, path: v} for k, v in tasks.items()]
    else:
        ext = os.path.splitext(os.path.basename(path))[0]
        if ext == "csv":
            tasks = pd.read_csv(path, sep=sep)
            if subset_row is not None:
                subset_row = read_subset_rows(subset_row)
                tasks = tasks.iloc[subset_row]
            tasks = tasks.to_dict(orient="records")
        else:
            tasks = [{key: path}]
    return tasks


def get_filename_from_str(string, n=100, has_signature=True, n_signature=8):
    name = string[:n]
    if has_signature:
        signature = get_signature_of_string(string, n_signature)
        name = "{}_{}".format(name, signature)
    return name


def read_subset_rows(string: str) -> List:
    """将1:2_3:4的字符串转化成整数索引列表，方便取子任务

    Args:
        string (str): _description_

    Returns:
        List: _description_
    """
    string = string.split("_")
    lst = []
    for s in string:
        if ":" in s:
            # 采用左闭、右闭方式
            start, end = [int(x) for x in s.split(":")]
            sub_lst = range(start, end + 1)
        else:
            sub_lst = [int(x) for x in s.split(",")]
        lst.extend(sub_lst)
    lst = sorted(set(lst))
    return lst


def fiss_tasks(tasks: List[Dict], task_fission_sep: str = "|") -> List[Dict]:
    """fiss tasks if task_fission_sep in value by product"""
    new_tasks = []
    for task in tasks:
        combination_fields = [
            k for k, v in task.items() if isinstance(v, str) and task_fission_sep in v
        ]
        if len(combination_fields) == 0:
            new_tasks.append(task)
            continue
        product_fields = [
            task[field].split(task_fission_sep) for field in combination_fields
        ]
        product_fields = list(product(*product_fields))
        # print("combination_fields", combination_fields)
        # print("product_fields", product_fields)
        for values in product_fields:
            task_cp = deepcopy(task)
            for i, field in enumerate(combination_fields):
                task_cp[field] = values[i]
            new_tasks.append(task_cp)
    return new_tasks
