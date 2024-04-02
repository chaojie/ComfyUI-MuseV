import os
import time
from typing import Literal, Union, List, Tuple
from tqdm import tqdm

from PIL import Image

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

import h5py
import torch
import numpy as np

from ...data.extract_feature.base_extract_feature import BaseFeatureExtractor
from ...data.emb.h5py_emb import save_value_with_h5py

from ..process.image_process import dynamic_crop_resize_image
from ..utils.data_type_util import convert_images

from .clip_vision_extractor import ImageClipVisionFeatureExtractorV2


class InsightFaceExtractor(BaseFeatureExtractor):
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
        model_name: str = "buffalo_l",
        allowed_modules: List[str] = ["detection", "recognition"],
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        need_align_face: bool = False,
    ):
        from insightface.app import FaceAnalysis

        super().__init__(device, dtype, name)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.extractor = FaceAnalysis(
            name=model_name,
            root=pretrained_model_name_or_path,
            allowed_modules=allowed_modules,
            providers=providers,
        )
        self.extractor.prepare(ctx_id=0, det_size=(640, 640))
        self.need_align_face = need_align_face

    def extract_images(
        self,
        data: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray],
        target_width: int = None,
        target_height: int = None,
        return_type: str = "numpy",
        input_rgb_order: str = "rgb",
    ) -> Union[np.ndarray, torch.Tensor]:
        data = convert_images(
            data,
            return_type="pil",
            input_rgb_order=input_rgb_order,
            return_rgb_order="bgr",
        )
        if target_height is not None and target_width is not None:
            data = [
                dynamic_crop_resize_image(
                    image,
                    target_height=target_height,
                    target_width=target_width,
                )
                for image in data
            ]
        data = [np.array(x.convert("RGB"))[:, :, ::-1] for x in data]
        with torch.no_grad():
            faces = [self.extractor.get(x) for x in data]

        emb = [self.get_target_emb(x) for x in faces]
        if self.need_align_face:
            from insightface.utils import face_align

            align_face_image = [
                face_align.norm_crop(x, landmark=faces[i][0].kps, image_size=224)
                for i, x in enumerate(data)
            ]
        else:
            align_face_image = None
        emb = np.concatenate(np.expand_dims(emb, axis=0), axis=0)
        if return_type == "torch":
            emb = torch.from_numpy(emb).to(device=self.device)
        return emb, align_face_image

    def get_target_emb(self, data):
        outputs = data[0]["embedding"]
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
        if self.need_align_face:
            align_face_images = []
        else:
            align_face_images = None
        with torch.no_grad():
            for i, (batch, batch_index) in enumerate(video_dataset):
                # TODO: 现阶段复用hugging face diffusers img2img pipeline中的抽取代码，
                # 由于该代码目前只支持Image的预处理，故先将numpy.ndarray转换成PIL.Image
                batch = [Image.fromarray(batch[b_i]) for b_i in range(len(batch))]
                emb, align_face_image = self.extract_images(
                    data=batch,
                    target_width=target_width,
                    target_height=target_height,
                    return_type=return_type,
                    input_rgb_order=input_rgb_order,
                )
                embs.append(emb)
                sample_indexs.extend(batch_index)
                if self.need_align_face:
                    align_face_images.append(align_face_image)
        sample_indexs = np.array(sample_indexs)
        if return_type == "numpy":
            embs = np.concatenate(embs, axis=0)
        elif return_type == "torch":
            embs = torch.concat(embs)
            sample_indexs = torch.from_numpy(sample_indexs)
        return sample_indexs, embs, align_face_images

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


class InsightFaceExtractorNormEmb(InsightFaceExtractor):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        name: str = None,
        device: str = "cpu",
        dtype=torch.float32,
        model_name: str = "buffalo_l",
        allowed_modules: List[str] = ["detection", "recognition"],
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        super().__init__(
            pretrained_model_name_or_path,
            name,
            device,
            dtype,
            model_name,
            allowed_modules,
            providers,
        )

    def get_target_emb(self, data):
        outputs = data[0].normed_embedding
        return outputs
