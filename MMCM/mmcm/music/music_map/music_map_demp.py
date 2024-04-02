from moviepy.editor import (
    ColorClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
)

from ...vision.video_map.video_lyric import render_lyric2video
from ...vision.video_map.video_writer import write_videoclip
from .music_map import MusicMap


def generate_music_map_videodemo(
    music_map: MusicMap,
    path: str,
    audio_path: str,
    render_lyric: bool = True,
    width: int = 360,
    height: int = 240,
    fps: int = 25,
    n_thread: int = 8,
    colors: list = [[51, 161, 201], [46, 139, 87]],
) -> None:
    """输入音乐谱面，生成对应的转场视频Demo，视频内容只是简单的颜色切换

    Args:
        music_map (MusicInfo): 待可视化的音乐谱面
        path (str): 可视化视频的存储路径
        audio_path (str): 音乐谱面对应的音频路径
        render_lyric (bool, optional): 是否渲染歌词，歌词在音乐谱面中. Defaults to True.
        width (int, optional): 可视化视频的宽. Defaults to 360.
        height (int, optional): 可视化视频的高. Defaults to 240.
        fps (int, optional): 可视化视频的fps. Defaults to 25.
        n_thread (int, optional): 可视化视频的写入线程数. Defaults to 8.
        colors (list, optional): 可视化的视频颜色. Defaults to [[51, 161, 201], [46, 139, 87]].
    """
    audio_clip = AudioFileClip(audio_path)
    video_clips = []
    size = (width, height)
    for i, clip in enumerate(music_map.clipseq):
        clip = ColorClip(
            size=size, color=colors[i % len(colors)], duration=clip.duration
        )
        video_clips.append(clip)
    video_clips = concatenate_videoclips(video_clips, method="compose")
    if render_lyric:
        video_clips = render_lyric2video(
            videoclip=video_clips,
            lyric=music_map,
            lyric_info_type="music_map",
        )
    video_clips = video_clips.set_audio(audio_clip)
    write_videoclip(
        video_clips,
        path=path,
        fps=fps,
        n_thread=n_thread,
    )
