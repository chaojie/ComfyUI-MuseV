from copy import deepcopy
import inspect
from typing import Any, Callable, Dict, List, Literal, Tuple, Union
import warnings
import os
import random

import h5py

from MuseVdiffusers.image_processor import VaeImageProcessor
import cv2
from einops import rearrange, repeat
import numpy as np
import torch
from torch import nn
from PIL import Image
import MuseVcontrolnet_aux
from MuseVdiffusers.models.controlnet import ControlNetModel
from MuseVdiffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from MuseVcontrolnet_aux.dwpose import draw_pose, pose2map, candidate2pose

from ..process.image_process import dynamic_crop_resize_image

from ..utils.data_type_util import convert_images
from ...data.emb.h5py_emb import save_value_with_h5py
from ...data.extract_feature.base_extract_feature import BaseFeatureExtractor
import json


def json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


import time


def controlnet_tile_processor(img, **kwargs):
    return img


class ControlnetProcessor(object):
    def __init__(
        self,
        detector_name: str,
        detector_id: str = None,
        filename: str = None,
        cache_dir: str = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        processor_params: Dict = None,
        processor_name: str = None,
    ) -> None:
        self.detector_name = detector_name
        self.detector_id = detector_id
        self.processor_name = processor_name
        if detector_name is None:
            self.processor = None
            self.processor_params = {}
            if isinstance(processor_name, str) and "tile" in processor_name:
                self.processor = controlnet_tile_processor
        else:
            processor_cls = controlnet_aux.__dict__[detector_name]
            processor_cls_argspec = inspect.getfullargspec(processor_cls.__init__)
            self.processor_params = (
                processor_params if processor_params is not None else {}
            )
            if not hasattr(processor_cls, "from_pretrained"):
                self.processor = processor_cls()
            else:
                self.processor = processor_cls.from_pretrained(
                    detector_id,
                    cache_dir=cache_dir,
                    filename=filename,
                    **self.processor_params,
                )
            if hasattr(self.processor, "to"):
                self.processor = self.processor.to(device=device)
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        data: Union[
            Image.Image, List[Image.Image], str, List[str], np.ndarray, torch.Tensor
        ],
        data_channel_order: str,
        target_width: int = None,
        target_height: int = None,
        return_type: Literal["pil", "np", "torch"] = "np",
        return_data_channel_order: str = "b h w c",
        processor_params: Dict = None,
        input_rgb_order: str = "rgb",
        return_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        # TODO： 目前采用二选一的方式，后续可以改进为增量更新
        processor_params = processor_params if processor_params is not None else {}
        data = convert_images(
            data,
            return_type="pil",
            input_rgb_order=input_rgb_order,
            return_rgb_order=return_rgb_order,
            data_channel_order=data_channel_order,
        )
        height, width = data[0].height, data[0].width
        if target_width is None:
            target_width = width
        if target_height is None:
            target_height = height

        data = [
            dynamic_crop_resize_image(
                image, target_height=target_height, target_width=target_width
            )
            for image in data
        ]
        if self.processor is not None:
            data = [self.processor(image, **processor_params) for image in data]

        # return_pose_only (bool): if true, only return pose keypoints in array format
        if "return_pose_only" in processor_params.keys():
            if (
                self.detector_name == "DWposeDetector"
                and processor_params["return_pose_only"]
            ):
                # (18, 2)
                # (1, 18)
                # (2, 21, 2)
                # (1, 68, 2)
                # j=json.dumps(data)
                # json_str = json.dumps(data, default=json_serializer)
                # return json_str
                # print(len(data))

                item_lsit = []
                for candidate, subset in data:
                    # candidate shape (1, 134, 2)
                    # subset          (1, 134)
                    # print(candidate.shape)
                    # print(subset.shape)
                    subset = np.expand_dims(subset, -1)
                    item = np.concatenate([candidate, subset], -1)
                    # print(item.shape)
                    max_num = 20
                    if item.shape[0] > max_num:
                        item = item[:max_num]

                    if item.shape[0] < max_num:
                        pad_num = max_num - item.shape[0]
                        item = np.pad(item, ((0, pad_num), (0, 0), (0, 0)))
                    # print(item.shape)
                    # print()

                    item_lsit.append(item)

                return np.stack(item_lsit, axis=0)  # b, num_candidates, 134, 3

        if return_type == "pil":
            return data
        data = np.stack([np.asarray(image) for image in data], axis=0)
        if return_data_channel_order != "b h w c":
            data = rearrange(data, "b h w c -> {}".format(return_data_channel_order))
        if return_type == "np":
            return data
        if return_type == "torch":
            data = torch.from_numpy(data)
            return data


class MultiControlnetProcessor(object):
    def __init__(self, processors: List[ControlnetProcessor]) -> None:
        self.processors = processors

    def __call__(
        self,
        data: Union[
            Image.Image, List[Image.Image], str, List[str], np.ndarray, torch.Tensor
        ],
        data_channel_order: str,
        target_width: int = None,
        target_height: int = None,
        return_type: Literal["pil", "np", "torch"] = "np",
        return_data_channel_order: str = "b h w c",
        processor_params: List[Dict] = None,
        input_rgb_order: str = "rgb",
        return_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        if processor_params is not None:
            assert isinstance(
                processor_params, list
            ), f"type of datas should be list, but given {type(datas)}"
            assert len(processor_params) == len(
                self.processors
            ), f"length of datas({len(processor_params)}) be same as of {len(self.processors)}"
        datas = [
            processor(
                data=data,
                data_channel_order=data_channel_order,
                target_height=target_height,
                target_width=target_width,
                return_type=return_type,
                return_data_channel_order=return_data_channel_order,
                input_rgb_order=input_rgb_order,
                processor_params=processor_params[i],
            )
            for i, processor in enumerate(self.processors)
        ]
        return datas


class ControlnetFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_path: str,
        detector_name: str,
        detector_id: str,
        device: str = "cpu",
        dtype=torch.float32,
        name: str = None,
        # /group/30065/users/public/muse/models/stable-diffusion-v1-5/vae/config.json
        vae_config_block_out_channels: int = 4,
        processor_params: Dict = None,
        filename=None,
        cache_dir: str = None,
    ):
        super().__init__(device, dtype, name)
        self.model_path = model_path
        self.processor = ControlnetProcessor(
            detector_name=detector_name,
            detector_id=detector_id,
            filename=filename,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )
        self.vae_scale_factor = 2 ** (vae_config_block_out_channels - 1)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self.controlnet = ControlNetModel.from_pretrained(
            model_path,
        ).to(device=device, dtype=dtype)
        self.detector_name = detector_name

    def emb_name(self, width, height):
        return "{}_w={}_h={}_emb".format(self.name, width, height)

    def prepare_image(
        self,
        image,  # b c t h w
        width,
        height,
    ):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.ndim == 5:
            image = rearrange(image, "b c t h w-> (b t) c h w")
        if height is None:
            height = image.shape[-2]
        if width is None:
            width = image.shape[-1]
        width, height = (
            x - x % self.control_image_processor.vae_scale_factor
            for x in (width, height)
        )
        image = image / 255.0
        # image = torch.nn.functional.interpolate(image, size=(height, width))
        do_normalize = self.control_image_processor.config.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.control_image_processor.normalize(image)
        return image

    def extract_images(
        self,
        data: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray],
        target_width: int = None,
        target_height: int = None,
        return_type: str = "numpy",
        data_channel_order: str = "b h w c",
        processor_params: Dict = None,
        input_rgb_order: str = "rgb",
        return_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        data = self.processor(
            data,
            data_channel_order=data_channel_order,
            target_height=target_height,
            target_width=target_width,
            return_type="torch",
            processor_params=processor_params,
            return_data_channel_order="b c h w",
            input_rgb_order=input_rgb_order,
            return_rgb_order=return_rgb_order,
        )

        # return_pose_only (bool): if true, only return pose keypoints in array format
        if "return_pose_only" in processor_params.keys():
            if (
                self.detector_name == "DWposeDetector"
                and processor_params["return_pose_only"]
            ):
                return data
        batch = self.prepare_image(image=data, width=target_width, height=target_height)

        with torch.no_grad():
            batch = batch.to(self.device, dtype=self.dtype)
            emb = self.controlnet.controlnet_cond_embedding(batch)
        if return_type == "numpy":
            emb = emb.cpu().numpy()

        return emb

    def extract_video(
        self,
        video_dataset,
        target_width: int = None,
        target_height: int = None,
        return_type: str = "numpy",
        processor_params: Dict = None,
        input_rgb_order: str = "rgb",
        return_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        embs = []
        sample_indexs = []
        with torch.no_grad():
            for i, (batch, batch_index) in enumerate(video_dataset):
                # print(f"============== extract img begin")
                # print(batch.shape)
                t0 = time.time()
                emb = self.extract_images(
                    data=batch,
                    target_width=target_width,
                    target_height=target_height,
                    return_type=return_type,
                    processor_params=processor_params,
                    input_rgb_order=input_rgb_order,
                    return_rgb_order=return_rgb_order,
                )
                torch.cuda.synchronize()
                t1 = time.time()

                # print(f"============== extract img end  TIME COST:{t1-t0}\n")

                embs.append(emb)
                sample_indexs.extend(batch_index)

        sample_indexs = np.array(sample_indexs)

        # return_pose_only (bool): if true, only return pose keypoints in array format
        if "return_pose_only" in processor_params.keys():
            if (
                self.detector_name == "DWposeDetector"
                and processor_params["return_pose_only"]
            ):
                embs = np.concatenate(embs, axis=0)
                return sample_indexs, embs

        if return_type == "numpy":
            embs = np.concatenate(embs, axis=0)
        elif return_type == "torch":
            embs = torch.concat(embs, dim=0)
            sample_indexs = torch.from_numpy(sample_indexs)
        return sample_indexs, embs

    def extract(
        self,
        data: Union[str, List[str]],
        data_type: Literal["image", "video"],
        return_type: str = "numpy",
        save_emb_path: str = None,
        save_type: str = "h5py",
        emb_key: str = "emb",
        sample_index_key: str = "sample_indexs",
        insert_name_to_key: bool = False,
        overwrite: bool = False,
        target_width: int = None,
        target_height: int = None,
        save_sample_index: bool = True,
        processor_params: Dict = None,
        input_rgb_order: str = "rgb",
        return_rgb_order: str = "rgb",
        **kwargs,
    ) -> Union[np.ndarray, torch.tensor]:
        if self.name is not None and insert_name_to_key:
            emb_key = f"{self.name}_{emb_key}"
            sample_index_key = f"{self.name}_{sample_index_key}"
        if save_emb_path is not None and os.path.exists(save_emb_path):
            with h5py.File(save_emb_path, "r") as f:
                if not overwrite and emb_key in f and sample_index_key in f:
                    return None

        if data_type == "image":
            emb = self.extract_images(
                data=data,
                return_type=return_type,
                target_height=target_height,
                target_width=target_width,
                processor_params=processor_params,
                input_rgb_order=input_rgb_order,
                return_rgb_order=return_rgb_order,
            )
            if save_emb_path is None:
                return emb
            else:
                raise NotImplementedError("save images emb")
        elif data_type == "video":
            sample_indexs, emb = self.extract_video(
                video_dataset=data,
                return_type=return_type,
                processor_params=processor_params,
                input_rgb_order=input_rgb_order,
                return_rgb_order=return_rgb_order,
                target_height=target_height,
                target_width=target_width,
                **kwargs,
            )
            if save_emb_path is None:
                return sample_indexs, emb
            else:
                if save_type == "h5py":
                    self.save_video_emb_with_h5py(
                        save_emb_path=save_emb_path,
                        emb=emb,
                        emb_key=emb_key,
                        sample_indexs=sample_indexs,
                        sample_index_key=sample_index_key,
                        save_sample_index=save_sample_index,
                        overwrite=overwrite,
                    )
                    return sample_indexs, emb
                else:
                    raise ValueError(f"only support save_type={save_type}")

    @staticmethod
    def save_video_emb_with_h5py(
        save_emb_path: str,
        emb: np.ndarray = None,
        emb_key: str = "emb",
        sample_indexs: np.ndarray = None,
        sample_index_key: str = "sample_indexs",
        overwrite: bool = False,
        save_sample_index: bool = True,
    ) -> h5py.File:
        save_value_with_h5py(save_emb_path, value=emb, key=emb_key, overwrite=overwrite)
        if save_sample_index:
            save_value_with_h5py(
                save_emb_path,
                value=sample_indexs,
                key=sample_index_key,
                overwrite=overwrite,
                dtype=np.uint32,
            )


def get_controlnet_params(
    controlnet_names: Union[
        Literal[
            "pose",
            "pose_body",
            "pose_hand",
            "pose_face",
            "pose_hand_body",
            "pose_hand_face",
            "pose_all",
            "dwpose",
            "canny",
            "hed",
            "hed_scribble",
            "depth",
            "pidi",
            "normal_bae",
            "lineart",
            "lineart_anime",
            "zoe",
            "sam",
            "mobile_sam",
            "leres",
            "content",
            "face_detector",
        ],
        List[str],
    ],
    detect_resolution: int = None,
    image_resolution: int = None,
    include_body: bool = False,
    include_hand: bool = False,
    include_face: bool = False,
    hand_and_face: bool = None,
) -> Dict:
    """通过简单 字符串参数就选择配置好的完整controlnet参数

    Args:
        controlnet_conds (Union[ Literal[ &quot;pose&quot;, &quot;canny&quot;, &quot;hed&quot;, &quot;hed_scribble&quot;, &quot;depth&quot;, &quot;pidi&quot;, &quot;normal_bae&quot;, &quot;lineart&quot;, &quot;lineart_anime&quot;, &quot;zoe&quot;, &quot;sam&quot;, &quot;mobile_sam&quot;, &quot;leres&quot;, &quot;content&quot;, &quot;face_detector&quot;, ], List[str], ]): _description_
        detect_resolution (int, optional): controlnet_aux图像处理需要的参数，尽量是64的整倍数. Defaults to None.
        image_resolution (int, optional): controlnet_aux图像处理需要的参数，尽量是64的整倍数. Defaults to None.
        include_body (bool, optional): controlnet 是否包含身体. Defaults to False.
        hand_and_face (bool, optional): pose controlnet 是否包含头和身体. Defaults to False.

    Returns:
        Dict: ControlnetProcessor需要的字典参数
    """
    controlnet_cond_maps = {
        "pose": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": include_body,
                "include_hand": include_hand,
                "include_face": include_face,
                "hand_and_face": hand_and_face,
            },
        },
        "pose_body": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": True,
                "include_hand": False,
                "include_face": False,
                "hand_and_face": False,
            },
        },
        "pose_hand": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": False,
                "include_hand": True,
                "include_face": False,
                "hand_and_face": False,
            },
        },
        "pose_face": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": False,
                "include_hand": False,
                "include_face": True,
                "hand_and_face": False,
            },
        },
        "pose_hand_body": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": True,
                "include_hand": True,
                "include_face": False,
                "hand_and_face": False,
            },
        },
        "pose_hand_face": {
            "middle": "pose",
            "detector_name": "OpenposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": False,
                "include_hand": True,
                "include_face": True,
                "hand_and_face": True,
            },
        },
        "dwpose": {
            "middle": "dwpose",
            "detector_name": "DWposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "dwpose_face": {
            "middle": "dwpose",
            "detector_name": "DWposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_hand": False,
                "include_body": False,
            },
        },
        "dwpose_hand": {
            "middle": "dwpose",
            "detector_name": "DWposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_face": False,
                "include_body": False,
            },
        },
        "dwpose_body": {
            "middle": "dwpose",
            "detector_name": "DWposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_face": False,
                "include_hand": False,
            },
        },
        "dwpose_body_hand": {
            "middle": "dwpose",
            "detector_name": "DWposeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_face": False,
                "include_hand": True,
                "include_body": True,
            },
        },
        "canny": {
            "middle": "canny",
            "detector_name": "CannyDetector",
            # "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_canny",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "tile": {
            "middle": "tile",
            "detector_name": None,
            "detector_id": None,
            "controlnet_model_path": "lllyasviel/control_v11f1e_sd15_tile",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "include_body": include_body,
                "hand_and_face": hand_and_face,
            },
        },
        # 隶属线条检测
        "hed": {
            "middle": "hed",
            "detector_name": "HEDdetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/sd-controlnet-hed",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "hed_scribble": {
            "middle": "hed",
            "detector_name": "HEDdetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_scribble",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "depth": {
            "middle": "depth",
            "detector_name": "MidasDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11f1p_sd15_depth",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "pidi": {
            "middle": "pidi",
            "detector_name": "PidiNetDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11f1p_sd15_depth",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "normal_bae": {
            "middle": "normal_bae",
            "detector_name": "NormalBaeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_normalbae",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "lineart": {
            "middle": "lineart",
            "detector_name": "LineartDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_lineart",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
                "coarse": True,
            },
        },
        "lineart_anime": {
            "middle": "lineart_anime",
            "detector_name": "LineartAnimeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11p_sd15s2_lineart_anime",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "zoe": {
            "middle": "zoe",
            "detector_name": "ZoeDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11f1p_sd15_depth",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "sam": {
            "middle": "sam",
            "detector_name": "SamDetector",
            "detector_id": "ybelkada/segment-anything",
            "processor_cls_params": {"subfolder": "checkpoints"},
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_seg",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "mobile_sam": {
            "middle": "mobile_sam",
            "detector_name": "SamDetector",
            "detector_id": "dhkim2810/MobileSAM",
            "processor_cls_params": {
                "subfolder": "checkpoints",
                "model_type": "vit_t",
                "filename": "mobile_sam.pt",
            },
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_seg",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "leres": {
            "middle": "leres",
            "detector_name": "LeresDetector",
            "detector_id": "lllyasviel/Annotators",
            "controlnet_model_path": "lllyasviel/control_v11f1p_sd15_depth",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        #  error
        "content": {
            "middle": "content",
            "detector_name": "ContentShuffleDetector",
            "controlnet_model_path": "lllyasviel/control_v11e_sd15_shuffle",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
        },
        "face_detector": {
            "middle": "face_detector",
            "detector_name": "MediapipeFaceDetector",
            "processor_params": {
                "detect_resolution": detect_resolution,
                "image_resolution": image_resolution,
            },
            "controlnet_model_path": "lllyasviel/control_v11p_sd15_openpose",
        },
    }

    def complete(dct):
        if "detector_id" not in dct:
            dct["detector_id"] = None
        if "processor_cls_params" not in dct:
            dct["processor_cls_params"] = None
        return dct

    if isinstance(controlnet_names, str):
        return complete(controlnet_cond_maps[controlnet_names])
    else:
        params = [complete(controlnet_cond_maps[name]) for name in controlnet_names]
        return params


def load_controlnet_model(
    controlnet_names: Union[str, List[str]],
    device: str,
    dtype=torch.dtype,
    need_controlnet_processor: bool = True,
    need_controlnet=True,
    detect_resolution: int = None,
    image_resolution: int = None,
    include_body: bool = False,
    include_face: bool = False,
    include_hand: bool = False,
    hand_and_face: bool = None,
) -> Tuple[nn.Module, Callable, Dict]:
    controlnet_params = get_controlnet_params(
        controlnet_names,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        include_body=include_body,
        include_face=include_face,
        hand_and_face=hand_and_face,
        include_hand=include_hand,
    )
    if need_controlnet_processor:
        if not isinstance(controlnet_params, list):
            controlnet_processor = ControlnetProcessor(
                detector_name=controlnet_params["detector_name"],
                detector_id=controlnet_params["detector_id"],
                processor_params=controlnet_params["processor_cls_params"],
                device=device,
                dtype=dtype,
                processor_name=controlnet_params["middle"],
            )
            processor_params = controlnet_params["processor_params"]
        else:
            controlnet_processor = MultiControlnetProcessor(
                [
                    ControlnetProcessor(
                        detector_name=controlnet_param["detector_name"],
                        detector_id=controlnet_param["detector_id"],
                        processor_params=controlnet_param["processor_cls_params"],
                        device=device,
                        dtype=dtype,
                        processor_name=controlnet_param["middle"],
                    )
                    for controlnet_param in controlnet_params
                ]
            )
            processor_params = [
                controlnet_param["processor_params"]
                for controlnet_param in controlnet_params
            ]
    else:
        controlnet_processor = None
        processor_params = None

    if need_controlnet:
        if isinstance(controlnet_params, List):
            # TODO: support MultiControlNetModel.save_pretrained str path
            controlnet = MultiControlNetModel(
                [
                    ControlNetModel.from_pretrained(d["controlnet_model_path"])
                    for d in controlnet_params
                ]
            )
        else:
            controlnet_model_path = controlnet_params["controlnet_model_path"]
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path)
        controlnet = controlnet.to(device=device, dtype=dtype)
    else:
        controlnet = None

    return controlnet, controlnet_processor, processor_params


def prepare_image(
    image,  # b c t h w
    image_processor: Callable,
    width=None,
    height=None,
    return_type: Literal["numpy", "torch"] = "numpy",
):
    if isinstance(image, List) and isinstance(image[0], str):
        raise NotImplementedError
    if isinstance(image, List) and isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 5:
        image = rearrange(image, "b c t h w-> (b t) c h w")
    if height is None:
        height = image.shape[-2]
    if width is None:
        width = image.shape[-1]
    width, height = (x - x % image_processor.vae_scale_factor for x in (width, height))
    if height != image.shape[-2] or width != image.shape[-1]:
        image = torch.nn.functional.interpolate(
            image, size=(height, width), mode="bilinear"
        )
    image = image.to(dtype=torch.float32) / 255.0
    do_normalize = image_processor.config.do_normalize
    if image.min() < 0:
        warnings.warn(
            "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
            f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
            FutureWarning,
        )
        do_normalize = False

    if do_normalize:
        image = image_processor.normalize(image)
    if return_type == "numpy":
        image = image.numpy()
    return image


class PoseKPs2ImgConverter(object):
    def __init__(
        self,
        target_width: int,
        target_height: int,
        num_candidates: int = 10,
        image_processor: Callable = None,
        include_body: bool = True,
        include_face: bool = False,
        hand_and_face: bool = None,
        include_hand: bool = True,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.num_candidates = num_candidates
        self.image_processor = image_processor
        self.include_body = include_body
        self.include_face = include_face
        self.hand_and_face = hand_and_face
        self.include_hand = include_hand

    def __call__(self, kps: np.array) -> Any:
        # draw pose
        # (b, max_num=10, 134, 3) last dim, x,y,score
        num_candidates = 0
        for idx_t in range(self.num_candidates):
            if np.sum(kps[:, idx_t, :, :]) == 0:
                num_candidates = idx_t
                break
        if num_candidates > 0:
            kps = kps[:, 0:num_candidates, :, :]
            candidate = kps[..., :2]
            subset = kps[..., 2]

            poses = [
                candidate2pose(
                    candidate[i],
                    subset[i],
                    include_body=self.include_body,
                    include_face=self.include_face,
                    hand_and_face=self.hand_and_face,
                    include_hand=self.include_hand,
                )
                for i in range(candidate.shape[0])
            ]
            pose_imgs = [
                pose2map(
                    pose,
                    self.target_height,
                    self.target_width,
                    min(self.target_height, self.target_width),
                    min(self.target_height, self.target_width),
                )
                for pose in poses
            ]
            pose_imgs = np.stack(pose_imgs, axis=0)  # b h w c
        else:
            pose_imgs = np.zeros(
                shape=(kps.shape[0], self.target_height, self.target_width, 3),
                dtype=np.uint8,
            )
        pose_imgs = rearrange(pose_imgs, "b h w c -> b c h w")

        if self.image_processor is not None:
            pose_imgs = prepare_image(
                image=pose_imgs,
                width=self.target_width,
                height=self.target_height,
                image_processor=self.image_processor,
                return_type="numpy",
            )

        return pose_imgs
