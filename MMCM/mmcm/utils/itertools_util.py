from typing import Any, List, Union, Sequence, Tuple

import numpy as np


def generate_sample_idxs(
    total: int,
    window_size: int,
    step: int,
    sample_rate: int = 1,
    drop_last: bool = False,
    max_num_per_window: int = None,
) -> List[List[int]]:
    """generate sample idxs list by given relate parameters

    Args:
        total (int): total num of sampling source
        window_size (int):
        step (int): _description_
        sample_rate (int, optional): _description_. Defaults to 1.
        drop_last (bool, optional): wthether drop the last, if not enough for window_size. Defaults to False.

    Returns:
        List[List[int]]: sample idx list
    """
    idxs = range(total)
    idxs = [idx for i, idx in enumerate(idxs) if i % sample_rate == 0]
    sample_idxs = []
    new_total = len(idxs)
    last_idx = new_total - 1
    window_start = 0
    while window_start < new_total:
        window_end = window_start + window_size
        window = idxs[window_start:window_end]
        if max_num_per_window is not None and len(window) > max_num_per_window:
            window = uniform_sample_subseq(
                window, max_num=max_num_per_window, need_index=False
            )
        if window_end > new_total and drop_last:
            break
        else:
            sample_idxs.append(window)
        window_start += step
    return sample_idxs


def overlap2step(overlap: Union[int, float], window_size: int) -> int:
    if isinstance(overlap, int):
        step = window_size - overlap
    elif isinstance(overlap, float):
        if overlap <= 0:
            raise ValueError(f"relative overlap should be > 0, but given{overlap}")
        overlap = int(overlap * window_size)
    else:
        raise ValueError(
            f"overlap only support int(>0） or float(>0), but given {overlap} type({type(overlap)})"
        )
    return step


def step2overlap(step: int, window_size: int) -> int:
    overlap = window_size - step
    return overlap


def uniform_sample_subseq(
    seq: Sequence, max_num: int, need_index: bool = False
) -> Union[Sequence, Tuple[Sequence, Sequence]]:
    n_seq = len(seq)
    sample_num = min(n_seq, max_num)
    if n_seq <= max_num:
        if need_index:
            return seq, list(range(n_seq))
        else:
            return seq
    idx = sorted(list(set(np.linspace(0, n_seq - 1, dtype=int))))
    subseq = [seq[i] for i in idx]
    if need_index:
        return subseq, idx
    else:
        return subseq


def convert_list_flat2nest(
    seq: Sequence,
    window: int,
) -> List[List[Any]]:
    n_seq = len(seq)
    n_lst = n_seq // window + int(n_seq % window > 0)
    res = [seq[i * window : (i + 1) * window] for i in range(n_lst)]
    return res
