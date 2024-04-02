import numpy as np
from sklearn.preprocessing import normalize, minmax_scale
from scipy.signal import savgol_filter

# TODO：待更新音乐谱面的类信息
from ...data.clip.clip_process import (
    complete_clipseq,
    find_idx_by_clip,
    insert_endclip,
    insert_startclip,
    reset_clipseq_id,
)

from .music_clip import Clip, ClipSeq
from .music_clip import MusicClipSeq
from .music_map import MusicMap


def generate_lyric_map(
    path: str, duration: float = None, gap_th: float = 2
) -> MusicClipSeq:
    """从歌词文件中生成音乐谱面

    Args:
        path (str): 歌词文件路径
        duration (float, optional): 歌词对应音频的总时长. Defaults to None.
        gap_th (float, optional): 歌词中间的空白部分是否融合到上一个片段中. Defaults to 3.

    Returns:
        MusicClipSeq: 以歌词文件生成的音乐谱面
    """
    from ..music_map.lyric_process import lyricfile2musicinfo

    lyric_info = lyricfile2musicinfo(path)
    lyric_info = MusicMap(lyric_info, duration=duration)
    clipseq = lyric_info.clipseq
    lyric_info.meta_info.duration = duration
    # set part of nonlyric as clip whose timepoint is 0
    for i in range(len(clipseq)):
        clipseq[i].timepoint_type = -1
    lyric_info.clipseq = complete_clipseq(
        clipseq=clipseq, duration=duration, gap_th=gap_th
    )
    return lyric_info


def insert_field_2_clipseq(clipseq: ClipSeq, reference: ClipSeq, field: str) -> ClipSeq:
    """将reference中每个clip的字段信息根据赋给clipseq中最近的clip

    Args:
        clipseq (ClipSeq): 目标clip序列
        reference (ClipSeq): 参考clip序列
        field (str): 目标字段

    Returns:
        ClipSeq: 更新目标字段新值后的clip序列
    """
    for i, clip in enumerate(clipseq):
        idx = find_idx_by_clip(reference, clip=clip)
        if idx is not None:
            if getattr(reference[idx], field) is not None:
                clipseq[i].__dict__[field] = getattr(reference[idx], field)
    return clipseq


def insert_rythm_2_clipseq(clipseq, reference):
    """参考MSS字段的结构信息设置rythm信息。目前策略非常简单，主歌(Vx)0.25，副歌(Cx)0.75，其他为None

    Args:
        clipseq (ClipSeq): 目标clip序列，设置rythm字段
        reference (ClipSeq): 参考clip序列，参考stage字段

    Returns:
        ClipSeq: 更新rythm字段新值后的clip序列
    """

    def stage2rythm(stage):
        if "V" in stage:
            return 0.25
        elif "C" in stage:
            return 0.75
        else:
            return None

    for i, clip in enumerate(clipseq):
        idx = find_idx_by_clip(reference, clip=clip)
        if idx is not None:
            if reference[idx].rythm is not None:
                clipseq[i].rythm = stage2rythm(reference[idx].stage)
    return clipseq


def insert_rythm_from_clip(clipseq: MusicClipSeq, beat: np.array) -> MusicClipSeq:
    """给MusicClipSeq中的每个Clip新增节奏信息。目前使用
        1. 单位时间内的歌词数量特征, 使用 min-max 归一化到 0 - 1 之间
        2. 单位时间内的关键点数量，目前使用beatnet,使用 min-max 归一化到 0 - 1 之间
        3. 对1、2中的特征相加，并根据歌曲结构不同进行加权
    Args:
        clipseq (MusicClipSeq): 待处理的 MusicClipSeq
        beat (np.array): beat检测结果，Nx2,，用于结算单位时间内的关键点数。
            1st column is time,
            2rd is type,
                0, end point
                1, strong beat
                2,3,4 weak beat

    Returns:
        MusicClipSeq: 新增 rythm 的 MusicClipSeq
    """
    mss_cofficient = {
        "intro": 1.0,
        "bridge": 1.0,
        "end": 0.8,
        "VA": 1.0,
        "VB": 1.0,
        "CA": 1.6,
        "CB": 1.6,
    }
    # text_num_per_second
    text_num_per_second_lst = [clip.tnps for clip in clipseq if clip.tnps != 0]
    common_tnps = np.min(text_num_per_second_lst)
    tnps = np.array([clip.tnps if clip.tnps != 0 else common_tnps for clip in clipseq])
    tnps = minmax_scale(tnps)
    # beat point _num_per_second
    beat_pnps = np.zeros(len(clipseq))
    for i, clip in enumerate(clipseq):
        time_start = clip.time_start
        time_end = clip.time_end
        target_beat = beat[(beat[:, 0] >= time_start) & (beat[:, 0] < time_end)]
        beat_pnps[i] = len(target_beat) / clip.duration
    beat_pnps = minmax_scale(beat_pnps)

    # cofficient
    cofficients = np.array(
        [
            mss_cofficient[clip.stage]
            if clip.stage in mss_cofficient and clip.stage is not None
            else 1.0
            for clip in clipseq
        ]
    )

    rythm = cofficients * (tnps + beat_pnps)
    rythm = minmax_scale(rythm)
    rythm = savgol_filter(rythm, window_length=5, polyorder=3)
    rythm = minmax_scale(rythm)
    for i, clip in enumerate(clipseq):
        clip.dynamic = rythm[i]
    return clipseq
