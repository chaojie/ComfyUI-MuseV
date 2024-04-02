from typing import Dict, Callable, List
import os

from ..utils.data_util import dict_has_keys, dict_get_keys
from .model_cfg import ModelCfg


def get_model_path(
    model_names: List[str],
    online_dir: str,
    offline_dir: str,
    download_func: Callable,
) -> Dict:
    """get model_path dict by model_name. If not existed, do download.

    Args:
        model_name (str): _description_
        online_dir (str): _description_
        offline_dir (str): _description_
        download_func (Callable): _description_

    Returns:
        Dict: _description_
    """
    if not dict_has_keys(ModelCfg, model_names):
        print("please set online model_path at least for {}".format(model_names))
        return
    else:
        model_basename_dct = dict_get_keys(ModelCfg, model_names)
        offline_path_dct = {}
        for k, v in model_basename_dct.items():
            offline_path = os.path.join(offline_dir, v)
            os.makedirs(os.path.dirname(offline_path), exist_ok=True)
            if not os.path.exists(offline_path):
                online_path = os.path.join(online_dir, v)
                print(
                    "starting downloading models from {} to".format(
                        online_path, offline_path
                    )
                )
                download_func(online_path, offline_dir)
            else:
                print("load offline model from {}".format(offline_path))
            offline_path_dct[k] = offline_path
        return offline_path_dct
