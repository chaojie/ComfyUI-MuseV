"""用于将 mediamap中的emb存储独立出去，仍处于开发中
"""
import logging

import numpy as np


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

__all__ = ["MediaMapEmb"]


class MediaMapEmb(object):
    def __init__(self, path: str) -> None:
        """
        OfflineEmb = {
            "overall_algo": Emb,  # 整个文件的Emb
            # 整个文件的多维度 Emb
            "theme": np.array,  # 主题，
            "emotion_algo":  np.array,  # 情绪，
            "semantic_algo":  np.array,  # 语义

            "clips_overall_algo":  np.array, n_clip x clip_emb
            "clips_emotion_algo":  np.array, n_clip x clip_emb
            "clips_semantic_algo":  np.array, n_clip x clip_emb
            "clips_theme_algo":  np.array, n_clip x clip_emb

            "scenes_overall_algo":  np.array, n_scenes x scene_emb
            "scenes_emotion_algo":  np.array, n_scenes x scene_emb
            "scenes_semantic_algo":  np.array, n_scenes x scene_emb
            "scenes_theme_algo": E np.arraymb, n_scenes x scene_emb
            # 片段可以是转场切分、MusicStage等, clips目前属于转场切分片段
            # 若后续需要新增段落分割，可以和clips同级新增 stage字段。

            "frames_overall_algo":  np.array, n_frames x frame_emb
            "frames_emotion_algo":  np.array, n_frames x frame_emb
            "frames_semantic_algo":  np.array, n_frames x frame_emb
            "frames_theme_algo":  np.array, n_frames x frame_emb
            "frames_objs": {
                "frame_id": {  #
                    "overall_algo":  np.array, n_objs x obj_emb
                    "emotion_algo":  np.array, n_objs x obj_emb
                    "semantic_algo":  np.array, n_objs x obj_emb
                    "theme_algo":  np.array, n_objs x obj_emb
                }
            }
            "roles_algo": {
                "roleid": np.array, n x obj_emb
            }
        }


        Args:
            path (str): hdf5 存储路径
        """
        self.path = path

    def get_value(self, key, idx=None):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.get_value(key)

    def get_media(self, factor, algo):
        return self.get_value(f"{factor}_{algo}")

    def get_clips(self, factor, algo, idx=None):
        return self.get_value(f"clips_{factor}_{algo}", idx=idx)

    def get_frames(self, factor, algo, idx=None):
        return self.get_value(f"frames_{factor}_{algo}", idx=idx)

    def get_frame_objs(self, frame_idx, factor, algo, idx=None):
        return self.get_value(["frames_objs", frame_idx, f"{factor}_{algo}"], idx=idx)

    def set_value(self, key, value, idx=None):
        raise NotImplementedError

    def set_media(self, factor, value, algo):
        self.set_value([f"{factor}_{algo}"], value)

    def set_clips(self, factor, value, algo, idx=None):
        self.set_value([f"clips_{factor}_{algo}"], value, idx=idx)

    def set_frames(self, factor, value, algo, idx=None):
        self.set_value([f"frames_{factor}_{algo}"], value)

    def set_frame_objs(self, frame_idx, factor, value, algo, idx=None):
        return self.set_value(
            ["frames_objs", frame_idx, f"{factor}_{algo}"], value, idx=idx
        )

    def set_roles(self, algo, value, idx=None):
        return self.set_value(f"roles_{algo}", value, idx=idx)

    def get_roles(self, algo, idx=None):
        return self.get_value(f"roles_{algo}", idx=idx)

    def __setitem__(self, key, value):
        self.set_value(self, key, value)


class MediaMapEmbProxy(MediaMapEmb):
    pass
