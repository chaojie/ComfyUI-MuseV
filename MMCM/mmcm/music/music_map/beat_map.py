import numpy as np

from librosa.core.audio import get_duration

from ...data.clip.clip_process import insert_endclip, insert_startclip

from .clip_process import filter_clipseq_target_point
from .music_clip import MusicClip, MusicClipSeq


def beatnet2TMEType(beat: np.array, duration: float) -> MusicClipSeq:
    """conver beatnet beat to tme beat type

    Args:
        beat (np.array): Nx2,
            1st column is time,
            2rd is type,
                0, end point
                1, strong beat
                2,3,4 weak beat
                -1 lyric
        duration (float): audio time length
    Returns:
        MusicClipSeq:
    """
    n = len(beat)
    beat = np.insert(beat, 0, 0, axis=0)
    beat = np.insert(beat, n + 1, [duration, 0], axis=0)
    clips = []
    for i in range(n + 1):
        beat_type = int(beat[i + 1, 1])
        clip = MusicClip(
            time_start=beat[i, 0],  # 开始时间
            duration=round(beat[i + 1, 0] - beat[i, 0], 3),  # 片段持续时间
            clipid=i,  # 片段序号，
            timepoint_type=beat_type,
        )
        clips.append(clip)
    clipseq = MusicClipSeq(clips=clips)
    return clipseq


def generate_beatseq_with_beatnet(audio_path: str) -> np.array:
    """使用beatnet生成beat序列

    Args:
        audio_path (str):
    Returns:
        np.array: beat序列 Nx2,
            1st column is time,
            2rd is type,
                0, end point
                1, strong beat
                2,3,4 weak beat
    """
    from BeatNet.BeatNet import BeatNet

    estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
    output = estimator.process(audio_path=audio_path)
    return output


def generate_music_map_with_beatnet(
    audio_path: str, target: list = [0, 1]
) -> MusicClipSeq:
    """使用beatnet生成beat MusicClipseq

    Args:
        audio_path (str):
        target (list, optional): 只保留相应的拍点. Defaults to [0, 1].

    Returns:
        MusicClipSeq: 返回的beat序列
        beat: np.array, 原始的beat检测结果
    """
    output = generate_beatseq_with_beatnet(audio_path)
    duration = get_duration(filename=audio_path)
    clipseq = beatnet2TMEType(output, duration)
    clipseq = insert_startclip(clipseq)
    clipseq = insert_endclip(clipseq, duration)
    clipseq = filter_clipseq_target_point(clipseq, target=target)
    return clipseq, output
