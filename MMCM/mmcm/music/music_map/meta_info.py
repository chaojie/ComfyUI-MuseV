from __future__ import annotations

from ...data import MetaInfo


class MusicMetaInfo(MetaInfo):
    def __init__(self, mediaid=None, media_name=None, media_duration=None, signature=None, media_path: str = None, media_map_path: str = None,
        singer=None,
        lyric_path=None,
        genre=None,
        language=None,
        start: float = None, end: float = None, ext=None, **kwargs):
        super().__init__(mediaid, media_name, media_duration, signature, media_path, media_map_path, start, end, ext, **kwargs)
        self.singer = singer
        self.genre = genre
        self.language = language
        self.lyric_path = lyric_path

    @classmethod
    def from_data(cls, data) -> MusicMetaInfo:
        return MusicMetaInfo(**data)