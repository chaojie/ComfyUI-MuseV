from typing import Callable, List, Union

from .clip import ClipSeq

from .clip_process import reset_clipseq_id


class ClipFilter(object):
    """clip滤波器，判断 Clip 是否符合标准

    Args:
        object (bool): 是否符合输入函数
    """

    def __init__(self, funcs: Union[Callable, List[Callable]], logic_func: Callable=all) -> None:
        """多个 clip 判断函数，通过 逻辑与、或当综合结果。

        Args:
            funcs (list of func): 列表判断函数
            logic_func (func, optional): all or any. Defaults to all.
        """
        self.funcs = funcs if isinstance(funcs, list) else [funcs]
        self.logic_func = logic_func

    def __call__(self, clip) -> bool:
        flag = [func(clip) for func in self.funcs]
        flag = self.logic_func(flag)
        return flag



# TODO
class ClipSeqFilter(object):
    def __init__(self, filter: Callable) -> None:
        self.filter = filter

    def __call__(self, clipseq: ClipSeq) -> ClipSeq:
        new_clipseq = []
        n_clipseq = len(clipseq)
        for i in range(n_clipseq):
            clip = clipseq[i]
            if self.filter(clip):
                new_clipseq.append(clip)
        new_clipseq = reset_clipseq_id(new_clipseq)
        # logger.debug("ClipSeqFilter: clipseq length before={}, after={}".format(n_clipseq, len(new_clipseq)))
        return new_clipseq
