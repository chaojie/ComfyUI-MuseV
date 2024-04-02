

from typing import List

from .music_map import MusicMap, MusicMapSeq


def load_music_map(
    music_map_paths,
    music_paths,
    emb_paths,
    start: float=None,
    end: None=None,
    target_stages: List[str] = None,
    **kwargs,
):
    """读取视频谱面，转化成MusicInfo。当 musicinfo_path_lst 为列表时，表示多歌曲

    Args:
        musicinfo_path_lst (str or [str]): 视频谱面路径文件列表
        music_path_lst (str or [str]): 视频文件路径文件列表，须与musicinfo_path_lst等长度


    Returns:
        MusicInfo: 视频谱面信息
    """
    dct ={
        "start": start,
        "end": end,
        "target_stages": target_stages,
    }
    if isinstance(music_map_paths, list):
        music_map = MusicMapSeq.from_json_paths(media_map_class=MusicMapSeq, media_paths=music_paths, media_map_paths=music_map_paths, emb_paths=emb_paths, **dct, **kwargs)
        if len(music_map) == 1:
            music_map = music_map[0]
    else:
        music_map = MusicMap.from_json_path(path=music_map_paths, emb_path=emb_paths, media_path=music_paths, **dct, **kwargs)
    return music_map