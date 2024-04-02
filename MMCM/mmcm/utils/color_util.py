from typing import List, Union, Tuple

class PolledColor(object):
    def __init__(self, colors: Union[List[Tuple[int, int, int]], List[Tuple[str, str, str]]]) -> None:
        """轮流返回候选颜色列表中的颜色

        Args:
            colors (list): 候选颜色列表
        """
        self.colors = colors
        self.cnt = 0
        self.n_color = len(colors)

    @property
    def color(self) -> Union[Tuple[int, int, int], Tuple[str, str, str]]:
        color = self.colors[self.cnt % self.n_color]
        self.cnt += 1
        return color
