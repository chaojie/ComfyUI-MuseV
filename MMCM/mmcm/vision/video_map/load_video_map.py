from .. import VideoMapSeq, VideoMetaInfo, VideoMap


def load_video_map(
    video_map_paths,
    video_paths,
    emb_paths,
    start: float = None,
    end: None = None,
    target_width_height_ratio: float = None,
    target_width: float = None,
    target_height: float = None,
    **kwargs,
):
    """读取视频谱面，转化成VideoInfo。当 videoinfo_path_lst 为列表时，表示多歌曲

    Args:
        videoinfo_path_lst (str or [str]): 视频谱面路径文件列表
        video_path_lst (str or [str]): 视频文件路径文件列表，须与videoinfo_path_lst等长度
        target_width_height_ratio (float): 目标视频宽高比
        target_width (int): 目标视频宽
        target_height (int): 目标视频高

    Returns:
        VideoInfo: 视频谱面信息
    """
    dct = {
        "start": start,
        "end": end,
        "target_width_height_ratio": target_width_height_ratio,
        "target_width": target_width,
        "target_height": target_height,
    }
    if isinstance(video_map_paths, list):
        video_map = VideoMapSeq.from_json_paths(
            media_map_class=VideoMap,
            media_paths=video_paths,
            media_map_paths=video_map_paths,
            emb_paths=emb_paths,
            **dct,
            **kwargs,
        )
        if len(video_map) == 1:
            video_map = video_map[0]
    else:
        video_map = VideoMap.from_json_path(
            path=video_map_paths,
            emb_path=emb_paths,
            media_path=video_paths,
            **dct,
            **kwargs,
        )
    return video_map
