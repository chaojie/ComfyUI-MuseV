import json
from typing import Tuple
import logging

from librosa.core.audio import get_duration
from moviepy.editor import CompositeVideoClip, concatenate_videoclips, TextClip
import moviepy as mvp

ignored_log = logging.getLogger("PIL")
ignored_log.setLevel(level=logging.INFO)

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def generate_lyric_video_from_music_map(
    music_map: dict,
    size=None,
    duration: float = None,
    fontsize: float = 50,
    padding: int = 0,
    gap_th: float = 2,
    font: str = "STXinwei",
):
    """从音乐谱面生成歌词 videoclip

    Args:
        music_map (dict): 音乐谱面，meta_info中必须含有歌词clip信息
        size (_type_, optional): _description_. Defaults to None.
        duration (float, optional): 歌词总时长. Defaults to None.
        fontsize (float, optional): 歌词字体大小. Defaults to 50.
        padding (int, optional): _description_. Defaults to 0.
        gap_th (float, optional): 补全歌词clip中的间隙部分. Defaults to 2.
        font (str, optional): 字体. Defaults to "STXinwei"，需要安装.

    Returns:
        moviepy.VideoClip: 生成的歌词视频
    """
    if isinstance(music_map, str):
        music_map = MusicInfo(music_map)
    lyric_clipseq = complete_clipseq(
        clipseq=music_map.meta_info.lyric, duration=duration, gap_th=gap_th
    )
    videoclips = []

    if music_map.meta_info.media_name is not None:
        media_name = music_map.meta_info.media_name
    else:
        media_name = ""
    if music_map.meta_info.singer is not None:
        singer = music_map.meta_info.singer
    else:
        singer = ""
    if music_map.meta_info.album is not None:
        album = music_map.meta_info.album
    else:
        album = ""
    title = "{} {} {}".format(album, media_name, singer)
    # if size is not None
    title_clip = TextClip(
        title,
        fontsize=int(fontsize * 1.1),
        color="white",
        font=font,
        stroke_width=2,
    )
    title_clip = title_clip.set_duration(3)
    for i, clip in enumerate(lyric_clipseq):
        time_start = clip.time_start
        duration = clip.duration
        if clip.text is not None:
            txt = clip.text
        else:
            txt = " "
        logger.debug(
            "render lyric, lyric={}, time_start={}, duration={}".format(
                txt, time_start, duration
            )
        )
        txt_clip = TextClip(
            txt, fontsize=fontsize, color="white", font=font, stroke_width=2
        )
        txt_clip = txt_clip.set_duration(duration)
        videoclips.append(txt_clip)
    videoclips = concatenate_videoclips(videoclips, method="compose")
    videoclips = CompositeVideoClip([videoclips, title_clip])
    videoclips.audio = None
    if duration is None:
        duration = lyric_clipseq[-1].time_start + lyric_clipseq[-1].duration
    # videoclips.set_duration(duration)
    return videoclips


def generate_lyric_video_from_lyric(
    path: str,
    audio_path: str = None,
    duration: float = None,
    size: Tuple = None,
    fontsize: int = None,
    padding: int = 0,
    font: str = "Courier",
):
    """从歌词文件中生成歌词视频

    Args:
        path (str): 歌词文件
        audio_path (str, optional): 对应的音频文件，主要用于提取音频总时长. Defaults to None.
        duration (float, optional): 歌曲总时长. Defaults to None.
        size (Tuple, optional): _description_. Defaults to None.
        fontsize (int, optional): 渲染的歌词字体大小. Defaults to None.
        padding (int, optional): _description_. Defaults to 0.

    Returns:
        moviepy. VideoClip: 渲染好的歌词视频
    """
    if audio_path is not None:
        duration = get_duration(audio_path)
    music_map = generate_lyric_map(path=path, duration=duration)
    clip = generate_lyric_video_from_music_map(
        music_map,
        size=size,
        duration=duration,
        padding=padding,
        fontsize=fontsize,
        font=font,
    )
    return clip


def render_lyric2video(
    videoclip,
    lyric: dict,
    lyric_info_type: str = "music_map",
    fontsize: int = 25,
    font: str = "Courier",
    audio_path: str = None,
    duration: float = None,
    padding: int = 0,
):
    """对视频进行歌词渲染

    Args:
        videoclip (moviepy.VideoClip): 待渲染的视频
        lyric (dict): 歌词信息，也可以是歌词路径
        lyric_info_type (str, optional): 歌词类型，可以是 qrc， 也可以是谱面. Defaults to "music_map".
        fontsize (int, optional): 渲染的歌词大小. Defaults to 25.
        audio_path (str, optional): 音频路径，主要提供一些必要信息. Defaults to None.
        duration (float, optional): 音频总时长. Defaults to None.
        padding (int, optional): _description_. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        moviepy.VideoClip: 渲染好歌词的视频文件
    """
    size = (videoclip.w, videoclip.h)

    if fontsize is None:
        fontsize = int(videoclip.w / 1280 * fontsize)
    if lyric_info_type == "lyric":
        lyric_clip = generate_lyric_video_from_lyric(
            lyric,
            size=size,
            fontsize=fontsize,
            font=font,
        )
    elif lyric_info_type == "music_map":
        lyric_clip = generate_lyric_video_from_music_map(
            lyric,
            size=size,
            fontsize=fontsize,
            font=font,
        )
    else:
        raise ValueError("not support {}".format(lyric_info_type))
    lyric_clip = lyric_clip.set_position(("center", "bottom"))
    lyric_video_clip = CompositeVideoClip([videoclip, lyric_clip], size=size)
    lyric_video_clip.audio = videoclip.audio
    logger.debug("lyric_clip: duration={}".format(lyric_clip.duration))
    return lyric_video_clip
