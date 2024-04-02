from typing import Tuple

import numpy as np

from .clip import ClipSeq


def stat_clipseq_duration(
    clipseq: ClipSeq,
) -> Tuple[np.array, np.array]:
    clip_duration = [clip.duration for clip in clipseq]
    (hist, bin_edges) = np.histogram(clip_duration)
    return hist, bin_edges
