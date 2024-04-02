import json
import os
import time
from typing import Literal, Optional, Union, List, Tuple
from tqdm import tqdm

from PIL import Image
from torch import nn
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    AutoProcessor,
)

import h5py
import torch
import numpy as np

from ...data.extract_feature.base_extract_feature import BaseFeatureExtractor
from ...data.emb.h5py_emb import save_value_with_h5py

from ..process.image_process import dynamic_crop_resize_image
from ..utils.data_type_util import convert_images


__all__ = [
    "ImageClipVisionFeatureExtractor",
    "ImageClipVisionFeatureExtractorV2",
    "ImageClipVisionFeatureExtractorV3",
    "ImageClipVisionFeatureExtractorV4",
    "VerstailSDLastHiddenState2ImageEmb",
    "OriginLastHiddenState2ImageEmbd",
    "OriginLastHiddenState2Poolout",
]


class ImageClipVisionFeatureExtractor(BaseFeatureExtractor):
    """选择clip的image_embeds，一张图像的输出特征是N，根据模型的选择可能是512、768、1024

    Args:
        BaseFeatureExtractor (_type_): _description_
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(device, dtype, name)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # 保持和 ipadapter 一致
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path
        ).to(device=device, dtype=dtype)
        # TODO: 存在多种初始化代码，待后续统一
        if os.path.isdir(pretrained_model_name_or_path):
            self.clip_image_processor = CLIPImageProcessor()
        else:
            self.clip_image_processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path
            )

    def extract_images(
        self,
        data: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray],
        target_width: int = None,
        target_height: int = None,
        return_type: str = "numpy",
        input_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        data = convert_images(data, return_type="pil", input_rgb_order=input_rgb_order)
        if target_height is not None and target_width is not None:
            data = [
                dynamic_crop_resize_image(
                    image,
                    target_height=target_height,
                    target_width=target_width,
                )
                for image in data
            ]

        with torch.no_grad():
            clip_image = self.clip_image_processor(
                images=data, return_tensors="pt"
            ).pixel_values
            emb = self.get_target_emb(
                clip_image.to(device=self.device, dtype=self.dtype)
            )
        if return_type == "numpy":
            emb = emb.cpu().numpy()
        return emb

    def get_target_emb(self, data):
        outputs = self.image_encoder(data).image_embeds
        return outputs

    def extract_video(
        self,
        video_dataset,
        target_width: int = None,
        target_height: int = None,
        return_type: str = "numpy",
        track_performance: bool = False,
        input_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        embs = []
        sample_indexs = []
        if track_performance:
            performance = {}
        with torch.no_grad():
            for i, (batch, batch_index) in enumerate(video_dataset):
                # TODO: 现阶段复用hugging face diffusers img2img pipeline中的抽取代码，
                # 由于该代码目前只支持Image的预处理，故先将numpy.ndarray转换成PIL.Image
                batch = [Image.fromarray(batch[b_i]) for b_i in range(len(batch))]
                emb = self.extract_images(
                    data=batch,
                    target_width=target_width,
                    target_height=target_height,
                    return_type=return_type,
                    input_rgb_order=input_rgb_order,
                )
                embs.append(emb)
                sample_indexs.extend(batch_index)
        sample_indexs = np.array(sample_indexs)
        if return_type == "numpy":
            embs = np.concatenate(embs, axis=0)
        elif return_type == "torch":
            embs = torch.concat(embs)
            sample_indexs = torch.from_numpy(sample_indexs)
        return sample_indexs, embs

    def extract(
        self,
        data: Union[str, List[str]],
        data_type: Literal["image", "video"],
        return_type: str = "numpy",
        save_emb_path: str = None,
        save_type: str = "h5py",
        emb_key: str = "image_embeds",
        sample_index_key: str = "sample_indexs",
        insert_name_to_key: bool = False,
        overwrite: bool = False,
        input_rgb_order: str = "rgb",
        save_sample_index: bool = True,
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
                input_rgb_order=input_rgb_order,
                **kwargs,
            )
            if save_emb_path is None:
                return emb
            else:
                raise NotImplementedError("save images emb")
        elif data_type == "video":
            sample_indexs, emb = self.extract_video(
                video_dataset=data,
                return_type=return_type,
                input_rgb_order=input_rgb_order,
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
                        overwrite=overwrite,
                        save_sample_index=save_sample_index,
                    )
                    return sample_indexs, emb
                else:
                    raise ValueError(f"only support save_type={save_type}")

    @staticmethod
    def save_images_emb_with_h5py(
        save_emb_path: str,
        emb: np.ndarray = None,
        emb_key: str = "image_embeds",
    ) -> h5py.File:
        save_value_with_h5py(save_emb_path, value=emb, key=emb_key)

    @staticmethod
    def save_video_emb_with_h5py(
        save_emb_path: str,
        emb: np.ndarray = None,
        emb_key: str = "image_embeds",
        sample_indexs: np.ndarray = None,
        sample_index_key: str = "sample_indexs",
        overwrite: bool = False,
        save_sample_index: bool = True,
    ) -> h5py.File:
        save_value_with_h5py(
            save_emb_path,
            value=emb,
            key=emb_key,
            overwrite=overwrite,
            dtype=np.float16,
        )
        if save_sample_index:
            save_value_with_h5py(
                save_emb_path,
                value=sample_indexs,
                key=sample_index_key,
                overwrite=overwrite,
                dtype=np.uint32,
            )


class ImageClipVisionFeatureExtractorV2(ImageClipVisionFeatureExtractor):
    """选择clip的 hidden_states[-2]，一张图像的输出特征是M*D，如257*1280，

    Args:
        BaseFeatureExtractor (_type_): _description_
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(pretrained_model_name_or_path, name, device, dtype)

    def get_target_emb(self, data):
        outputs = self.image_encoder(data, output_hidden_states=True).hidden_states[-2]
        return outputs


class ImageClipVisionFeatureExtractorV3(ImageClipVisionFeatureExtractor):
    """选择clip的 hidden_states[-2]，一张图像的输出特征是M*D，如257*1280，

    Args:
        BaseFeatureExtractor (_type_): _description_
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(pretrained_model_name_or_path, name, device, dtype)

    def get_target_emb(self, data):
        outputs = self.image_encoder(data, output_hidden_states=True).last_hidden_state
        return outputs


class ImageClipVisionFeatureExtractorV4(ImageClipVisionFeatureExtractor):
    """
    参考 https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/versatile_diffusion/pipeline_versatile_diffusion_image_variation.py#L114

    Args:
        BaseFeatureExtractor (_type_): _description_
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(pretrained_model_name_or_path, name, device, dtype)

    def get_target_emb(self, data):
        encoder_output = self.image_encoder(data, output_hidden_states=True)
        embeds = self.image_encoder.vision_model.post_layernorm(
            encoder_output.last_hidden_state
        )
        embeds = self.image_encoder.visual_projection(embeds)
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds


class OriginLastHiddenState2Poolout(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        projection_dim: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.visual_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def load_state_dict_from_pretrained(self, pretrained_model_name_or_path):
        model_pretrained = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        post_layernorm_params = {
            k.replace("vision_model.post_layernorm.", ""): v
            for k, v in model_pretrained.items()
            if "vision_model.post_layernorm." in k
        }
        self.post_layernorm.load_state_dict(post_layernorm_params)
        visual_projection_params = {
            k.replace("visual_projection.", ""): v
            for k, v in model_pretrained.items()
            if "visual_projection." in k
        }
        self.visual_projection.load_state_dict(visual_projection_params)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        cfg_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(cfg_path, "r") as f:
            config = json.load(f)
        model = cls(
            hidden_size=config["hidden_size"],
            projection_dim=config["projection_dim"],
            layer_norm_eps=config["layer_norm_eps"],
        )
        model.load_state_dict_from_pretrained(pretrained_model_name_or_path)
        return model

    def forward(self, data):
        last_hidden_state = data
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        # image_embeds = self.visual_projection(pooled_output)
        return pooled_output


class OriginLastHiddenState2ImageEmbd(OriginLastHiddenState2Poolout):
    def __init__(self, hidden_size: int, projection_dim: int, layer_norm_eps: float):
        super().__init__(hidden_size, projection_dim, layer_norm_eps)

    def forward(self, data):
        pooled_output = super().forward(data)
        image_embeds = self.visual_projection(pooled_output)
        return image_embeds


class VerstailSDLastHiddenState2ImageEmb(OriginLastHiddenState2ImageEmbd):
    def __init__(self, hidden_size: int, projection_dim: int, layer_norm_eps: float):
        super().__init__(hidden_size, projection_dim, layer_norm_eps)

    def forward(self, data):
        embeds = self.post_layernorm(data)
        embeds = self.visual_projection(embeds)
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds
