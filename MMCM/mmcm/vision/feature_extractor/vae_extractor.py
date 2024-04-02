import sys
from multiprocessing.pool import Pool
import os
import logging
from typing import Union, List, Tuple, Literal

import torch
import numpy as np
import pandas as pd
import h5py
import diffusers
from MuseVdiffusers import AutoencoderKL
from MuseVdiffusers.image_processor import VaeImageProcessor
from einops import rearrange
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from ...data.extract_feature.base_extract_feature import BaseFeatureExtractor
from ...data.emb.h5py_emb import save_value_with_h5py

from ..process.image_process import dynamic_resize_image, dynamic_crop_resize_image
from ..utils.data_type_util import convert_images


class VAEFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        super().__init__(device, dtype, name)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        vae.requires_grad_(False)
        self.vae = vae.to(device=device, dtype=dtype)
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

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
        batch = self.image_processor.preprocess(data).to(
            device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            # print("batch", batch.shape, batch.dtype, batch.device, self.vae.device)
            emb = self.vae.encoder(batch)
            quant_emb = self.vae.quant_conv(emb)
        if return_type == "numpy":
            emb = emb.cpu().numpy()
            quant_emb = quant_emb.cpu().numpy()
        return emb, quant_emb

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
        quant_embs = []
        sample_indexs = []
        if track_performance:
            performance = {}
        with torch.no_grad():
            for i, (batch, batch_index) in enumerate(video_dataset):
                # TODO: 现阶段复用hugging face diffusers img2img pipeline中的抽取代码，
                # 由于该代码目前只支持Image的预处理，故先将numpy.ndarray转换成PIL.Image
                batch = [Image.fromarray(batch[b_i]) for b_i in range(len(batch))]
                emb, quant_emb = self.extract_images(
                    data=batch,
                    target_width=target_width,
                    target_height=target_height,
                    return_type=return_type,
                    input_rgb_order=input_rgb_order,
                )
                embs.append(emb)
                quant_embs.append(quant_emb)
                sample_indexs.extend(batch_index)

        sample_indexs = np.array(sample_indexs)
        if return_type == "numpy":
            embs = np.concatenate(embs, axis=0)
            quant_embs = np.concatenate(quant_embs, axis=0)
        elif return_type == "torch":
            embs = torch.concat(embs)
            quant_embs = torch.concat(quant_embs)
            sample_indexs = torch.from_numpy(sample_indexs)
        return sample_indexs, embs, quant_embs

    def extract(
        self,
        data: Union[str, List[str]],
        data_type: Literal["image", "video"],
        return_type: str = "numpy",
        save_emb_path: str = None,
        save_type: str = "h5py",
        emb_key: str = "encoder_emb",
        quant_emb_key: str = "encoder_quant_emb",
        sample_index_key: str = "sample_indexs",
        insert_name_to_key: bool = False,
        overwrite: bool = False,
        save_sample_index: bool = True,
        input_rgb_order: str = "rgb",
        **kwargs,
    ) -> Union[np.ndarray, torch.tensor]:
        if self.name is not None and insert_name_to_key:
            emb_key = f"{self.name}_{emb_key}"
            quant_emb_key = f"{self.name}_{quant_emb_key}"
            sample_index_key = f"{self.name}_{sample_index_key}"
        if save_emb_path is not None and os.path.exists(save_emb_path):
            with h5py.File(save_emb_path, "r") as f:
                if (
                    not overwrite
                    and emb_key in f
                    and quant_emb_key in f
                    and sample_index_key in f
                ):
                    return None

        if data_type == "image":
            emb, quant_emb = self.extract_images(
                data=data,
                return_type=return_type,
                input_rgb_order=input_rgb_order,
                **kwargs,
            )
            if save_emb_path is None:
                return emb, quant_emb
            else:
                raise NotImplementedError("save images emb")
        elif data_type == "video":
            sample_indexs, emb, quant_emb = self.extract_video(
                video_dataset=data,
                return_type=return_type,
                input_rgb_order=input_rgb_order,
                **kwargs,
            )
            if save_emb_path is None:
                return sample_indexs, emb, quant_emb
            else:
                if save_type == "h5py":
                    self.save_video_emb_with_h5py(
                        save_emb_path=save_emb_path,
                        emb=emb,
                        emb_key=emb_key,
                        quant_emb=quant_emb,
                        quant_emb_key=quant_emb_key,
                        sample_indexs=sample_indexs,
                        sample_index_key=sample_index_key,
                        save_sample_index=save_sample_index,
                        overwrite=overwrite,
                    )
                    return sample_indexs, emb, quant_emb
                else:
                    raise ValueError(f"only support save_type={save_type}")

    @staticmethod
    def save_images_emb_with_h5py(
        save_emb_path: str,
        emb: np.ndarray = None,
        emb_key: str = "encoder_emb",
        quant_emb: np.ndarray = None,
        quant_emb_key: str = "encoder_quant_emb",
    ) -> h5py.File:
        save_value_with_h5py(save_emb_path, value=emb, key=emb_key)
        save_value_with_h5py(save_emb_path, value=quant_emb, key=quant_emb_key)

    @staticmethod
    def save_video_emb_with_h5py(
        save_emb_path: str,
        emb: np.ndarray = None,
        emb_key: str = "encoder_emb",
        quant_emb: np.ndarray = None,
        quant_emb_key: str = "encoder_quant_emb",
        sample_indexs: np.ndarray = None,
        sample_index_key: str = "sample_indexs",
        overwrite: bool = False,
        save_sample_index: bool = True,
    ) -> h5py.File:
        # save_value_with_h5py(save_emb_path, value=emb, key=emb_key, overwrite=overwrite)
        if save_sample_index:
            save_value_with_h5py(
                save_emb_path,
                value=quant_emb,
                key=quant_emb_key,
                overwrite=overwrite,
                dtype=np.float16,
            )
            save_value_with_h5py(
                save_emb_path,
                value=sample_indexs,
                key=sample_index_key,
                overwrite=overwrite,
                dtype=np.uint32,
            )
