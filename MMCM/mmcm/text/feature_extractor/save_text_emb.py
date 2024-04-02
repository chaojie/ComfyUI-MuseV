from typing import Union

import h5py
import numpy as np

from ...data.emb.h5py_emb import save_value_with_h5py


def save_text_emb_with_h5py(
    path: str,
    emb: Union[np.ndarray, None] = None,
    text_emb_key: str = None,
    text: str = None,
    text_key: str = "text",
    text_tuple_length: int = 20,
    text_index: int = 0,
) -> None:
    if emb is not None:
        save_value_with_h5py(path, value=emb, key=text_emb_key)
    if text is not None:
        save_value_with_h5py(
            path,
            key=text_key,
            value=text,
            shape=(text_tuple_length,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            idx=text_index,
        )
