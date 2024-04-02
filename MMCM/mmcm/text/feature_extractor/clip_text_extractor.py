import sys
from multiprocessing.pool import Pool
import os
import logging
from typing import Union, List, Tuple

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

from .save_text_emb import save_text_emb_with_h5py


class ClipTextFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "cpu",
        dtype: torch.dtype = None,
        name: str = "CLIPEncoderLayer",
    ):
        super().__init__(device, dtype, name)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder.to(device=device, dtype=dtype)

    def extract(
        self,
        text: Union[str, List[str]],
        return_type: str = "numpy",
        save_emb_path: str = None,
        save_type: str = "h5py",
        text_emb_key: str = None,
        text_key: str = "text",
        text_tuple_length: int = 20,
        text_index: int = 0,
        insert_name_to_key: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if text_emb_key is not None:
            text_emb_key = f"{text_emb_key}_{text_index}"
        if self.name is not None and insert_name_to_key:
            if text_emb_key is not None:
                text_emb_key = f"{self.name}_{text_emb_key}"
        text_inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None
        # transformers.modeling_outputs.BaseModelOutputWithPooling
        # 'last_hidden_state', 'pooler_output'
        # we choose the first
        print()
        text_embeds = self.text_encoder(
            text_input_ids.to(device=self.device),
            attention_mask=attention_mask,
        )[0]

        if return_type == "numpy":
            text_embeds = text_embeds.cpu().numpy()
        if save_emb_path is None:
            return text_embeds
        else:
            if save_type == "h5py":
                save_text_emb_with_h5py(
                    path=save_emb_path,
                    emb=text_embeds,
                    text_emb_key=text_emb_key,
                    text=text,
                    text_key=text_key,
                    text_tuple_length=text_tuple_length,
                    text_index=text_index,
                )
                return text_embeds
