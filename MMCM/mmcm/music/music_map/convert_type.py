from ...data.clip.clip_process import (
    insert_startclip,
    insert_endclip,
    reset_clipseq_id,
)

from .music_clip import MusicClip, MusicClipSeq


def read_osu_hitobjs(path: str) -> list:
    """读取osu的音游谱面

    Args:
        path (str): 谱面低质

    Returns:
        list: 只包含HitObjects的行字符串信息
    """
    lines = []
    is_hit_info_start = False
    with open(path, "r") as f:
        for line in f:
            if is_hit_info_start:
                lines.append(line.strip())
            if "[HitObjects]" in line:
                is_hit_info_start = True
    return lines


def osu2itech(src: list, duration: float = None) -> MusicClipSeq:
    """将osu的音游谱面转换为我们的目标格式

    Args:
        src (list): 音游谱面路径或者是读取的目标行字符串列表
        duration (float, optional): 歌曲长度. Defaults to None.

    Returns:
        MusicClipSeq: 音乐片段序列
    """
    if isinstance(src, str):
        src = read_osu_hitobjs(src)
    timepoints = [float(line.split(",")[2]) for line in src]
    clips = []
    for i in range(len(timepoints) - 1):
        clip = MusicClip(
            time_start=round(timepoints[i] / 1000, 3),
            timepoint_type=0,
            duration=round((timepoints[i + 1] - timepoints[i]) / 1000, 3),
            clipid=i,
        )
        clips.append(clip)
    if len(clips) > 0:
        clips = insert_startclip(clips)
        if duration is not None:
            clips = insert_endclip(clips, duration=duration)
        clips = reset_clipseq_id(clips)
    return MusicClipSeq(clips)
