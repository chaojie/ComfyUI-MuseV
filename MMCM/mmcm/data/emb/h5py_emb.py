from typing import Union, List
import logging

import h5py
import numpy as np

from .emb import MediaMapEmb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

__all__ = ["H5pyMediaMapEmb", "save_value_with_h5py"]


def save_value_with_h5py(
    path: str,
    value: Union[np.ndarray, None],
    key: str,
    idx: Union[int, List[int]] = None,
    dtype=None,
    shape=None,
    overwrite: bool = False,
):
    with h5py.File(path, "a") as f:
        if dtype is None:
            dtype = value.dtype
        if shape is None:
            shape = value.shape
        del_key = False
        if key in f:
            if overwrite:
                del_key = True
            if f[key].dtype != h5py.special_dtype(vlen=str):
                if f[key].shape != value.shape:
                    del_key = True
            if del_key:
                del f[key]
        if key not in f:
            f.create_dataset(key, shape=shape, dtype=dtype)
        if idx is None:
            f[key][...] = value
        else:
            f[key][idx] = value


class H5pyMediaMapEmb(MediaMapEmb):
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
            "frames_objs_algo": {
                "frame_id_algo": {  #
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
        super().__init__(path)
        # 待优化支持 with open 的方式来读写
        self.f = h5py.File(path, "a")

    def _keys_index(self, key):
        if not isinstance(key, list):
            key = [key]
        key = "/".join([str(x) for x in key if x is not None])
        return key

    def get_value(self, key, idx=None):
        new_key = self._keys_index(key)
        if idx is None:
            data = np.array(self.f[new_key])
        else:
            data = np.array(self.f[new_key][idx])
        return data

    def set_value(self, key, value, idx=None):
        new_key = self._keys_index(key)
        if new_key not in self.f:
            self.f.create_dataset(new_key, shape=value.shape, dtype=value.dtype)
        if idx is None:
            self.f[new_key][...] = value
        else:
            self.f[new_key][idx] = value

    def close(self):
        self.f.close()


class H5pyMediaMapEmbProxy(H5pyMediaMapEmb):
    pass
