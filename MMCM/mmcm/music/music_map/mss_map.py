import logging

from .music_clip import MusicClip, MusicClipSeq
from .music_map import MusicMap
from ...data.clip.clip_process import find_idx_by_time

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def insert_mss_2_clipseq(
    clipseq: MusicClipSeq, mss_clipseq: MusicClipSeq
) -> MusicClipSeq:
    """将mss中的结构字段信息赋予到目标clipseq中的最近clip

    Args:
        clipseq (ClipSeq): 目标clip序列
        reference (ClipSeq): 参考clip序列
        field (str): 目标字段

    Returns:
        ClipSeq: 更新目标字段新值后的clip序列
    """
    for i, clip in enumerate(clipseq):
        idx = find_idx_by_time(mss_clipseq, clip.time_start)
        if idx is not None:
            clipseq[i].stage = mss_clipseq[idx].stage
        else:
            clipseq[i].stage = "unknow"
    return clipseq


def get_mss_musicinfo(songid: str) -> MusicMap:
    """通过调用media_data中的接口 获取天琴实验室的歌曲结构信息

    Args:
        songid (str): 歌词id

    Returns:
        MusicMap: mss结构信息生成的音乐谱面
    """
    try:
        from media_data.oi.tianqin_database import get_mss

        mss = get_mss(songid=songid)
    except Exception as e:
        logger.warning("get mss failed, mss={}".format(songid))
        logger.exception(e)
        mss = None
    mss_musicinfo = MusicMap(mss) if mss is not None else None
    return mss_musicinfo


def merge_mss(musicinfo: MusicMap, mss: MusicMap) -> MusicMap:
    """融合mss音乐谱面到目标音乐谱面

    Args:
        musicinfo (MusicMap): 目标音乐谱面
        mss (MusicMap): 待融合的mss音乐谱面

    Returns:
        MusicMap: 融合后的音乐谱面
    """
    musicinfo.meta_info.bpm = mss.meta_info.bpm
    if len(mss.clipseq) > 0:
        musicinfo.clipseq = insert_mss_2_clipseq(musicinfo.clipseq, mss.clipseq)
    return musicinfo


def generate_mss_from_lyric(lyrics: list, audio_duration: float, th=8) -> MusicClipSeq:
    # "intro", "VA", "CA", "bridge", "VB", "CB", "end"]
    mss = []
    n_lyric = len(lyrics)
    for lyric_idx, line_lyric_dct in enumerate(lyrics):
        time_start = line_lyric_dct["time_start"]
        duration = line_lyric_dct["duration"]
        time_end = time_start + duration
        # text = line_lyric_dct["text"]
        if lyric_idx == 0:
            sub_mss = {
                "stage": "intro",
                "time_start": 0,
                "duration": time_start,
            }
            mss.append(sub_mss)
            continue
        if lyric_idx == n_lyric - 1:
            sub_mss = {
                "stage": "end",
                "time_start": time_end,
                "duration": audio_duration - time_end,
            }
            mss.append(sub_mss)
            continue

        if lyrics[lyric_idx + 1]["time_start"] - time_end >= th:
            sub_mss = {
                "stage": "bridge",
                "time_start": time_end,
                "duration": lyrics[lyric_idx + 1]["time_start"] - time_end,
            }
            mss.append(sub_mss)
    mss_lyric = []
    for sub_idx, sub_mss in enumerate(mss):
        if sub_idx == len(mss) - 1:
            continue
        time_end = sub_mss["time_start"] + sub_mss["duration"]
        next_time_start = mss[sub_idx + 1]["time_start"]
        if next_time_start - time_end > 0.1:
            mss_lyric.append(
                {
                    "stage": "lyric",
                    "time_start": time_end,
                    "duration": next_time_start - time_end,
                }
            )
    mss.extend(mss_lyric)
    mss = sorted(mss, key=lambda x: x["time_start"])
    mss = MusicClipSeq(mss)
    return mss


def refine_mss_info_from_tianqin(
    mss_info: MusicMap, lyricseq: MusicClipSeq
) -> MusicMap:
    """优化天琴的歌曲结信息,
    优化前：天琴歌曲结构里面只有每句歌词和结构信息，时间前后不连续，对于整首歌去时间结构不完备。
    优化后：增加intro,bridge,end，将相近的结构信息合并，时间前后连续，时间完备

    Args:
        mss_info (MusicMap): 天琴歌曲结构
        lyricseq (ClipSeq): 原始歌曲信息，用于计算Intro,bridge,end。其实也可以从mss_info中获取。

    Returns:
        MusicMap: 优化后的歌曲结构信息
    """
    lyric_mss_clipseq = generate_mss_from_lyric(
        lyricseq, audio_duration=mss_info.meta_info.duration
    )
    new_mss_clipseq = []
    # lyric_mss_dct = lyric_mss_clipseq.to_dct()
    # mss_dct = mss_info.clipseq.to_dct()
    for l_clip_idx, lyric_clip in enumerate(lyric_mss_clipseq):
        if lyric_clip.stage != "lyric":
            new_mss_clipseq.append(lyric_clip)
        else:
            new_clip_time_start = lyric_clip.time_start
            last_stage = "ANewClipStart"
            for clip_idx, clip in enumerate(mss_info.clipseq):
                if clip.time_start < new_clip_time_start:
                    continue
                if (
                    clip.time_start >= lyric_mss_clipseq[l_clip_idx + 1].time_start
                    or clip_idx == len(mss_info.clipseq) - 1
                ):
                    if clip.time_start >= lyric_mss_clipseq[l_clip_idx + 1].time_start:
                        stage = last_stage
                    # 像偶阵雨这首歌最后一个歌词段落 只有一句歌词
                    if clip_idx == len(mss_info.clipseq) - 1:
                        stage = clip.stage
                    new_clip_time_end = lyric_mss_clipseq[l_clip_idx + 1].time_start
                    new_stage_clip = {
                        "time_start": new_clip_time_start,
                        "duration": new_clip_time_end - new_clip_time_start,
                        "stage": stage,
                    }
                    new_mss_clipseq.append(MusicClip(**new_stage_clip))
                    new_clip_time_start = new_clip_time_end
                    last_stage = clip.stage
                    break
                if clip.stage != last_stage:
                    if last_stage == "ANewClipStart":
                        last_stage = clip.stage
                        continue
                    new_clip_time_end = mss_info.clipseq[clip_idx].time_start
                    new_stage_clip = {
                        "time_start": new_clip_time_start,
                        "duration": new_clip_time_end - new_clip_time_start,
                        "stage": last_stage,
                    }
                    new_mss_clipseq.append(MusicClip(**new_stage_clip))
                    new_clip_time_start = new_clip_time_end
                    last_stage = clip.stage
    new_mss_clipseq = MusicClipSeq(sorted(new_mss_clipseq, key=lambda x: x.time_start))
    mss_info.clipseq = new_mss_clipseq
    return mss_info
