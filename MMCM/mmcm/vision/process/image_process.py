from einops import rearrange
import requests
from io import BytesIO
from typing import Literal, Union
import math

from PIL import Image

import numpy as np
from MuseVdiffusers.utils import load_image
import cv2
import torch

from mmcm.vision.utils.data_type_util import convert_images
from transformers.models.clip.image_processing_clip import to_numpy_array
from ..utils.vision_util import round_up_to_even


def get_image_from_input(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        if "http" in image:
            image = BytesIO(requests.get(image).content)
            image = Image.open(image).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    assert type(image) == Image.Image
    return image


def dynamic_resize_image(
    image: Image.Image,
    target_height: int,
    target_width: int,
    image_max_length: int = 910,
) -> Image.Image:
    """对图像进行预处理，目前会将短边resize到目标长度，同时限制长边长度

    Args:
        image (Image.Image): _description_
        target_height (int): _description_
        target_width (int): _description_
        image_max_length (int): _description_

    Returns:
        Image.Image: _description_
    """
    w, h = image.size
    if w > h:
        target_width = min(math.ceil(w * target_height / h), image_max_length)
        target_height = math.ceil(target_width / w * h)
    else:
        target_height = min(math.ceil(h * target_width / w), image_max_length)
        target_width = math.ceil(target_height / h * w)
    target_width = round_up_to_even(target_width)
    target_height = round_up_to_even(target_height)
    image = image.resize((target_width, target_height))
    return image


def dynamic_crop_resize_image(
    image: Image.Image,
    target_height: int,
    target_width: int,
    resample=None,
) -> Image.Image:
    """获取图像有效部分，并resize到对应目标宽度和高度。
        如果图像宽高比大于 target_width / target_height，则保留全部高，截取宽的中心部位；
        如果图像宽高比小于 target_width / target_height，则保留全部宽，截取高的中心部位；
        最后，将截取的图像resize到目标宽高
    Args:
        image (Image.Image): 输入图像
        target_height (int): 目标高
        target_width (int): 目标宽

    Returns:
        Image.Image: 动态截取、resize生成的图像
    """
    w, h = image.size
    image_width_heigt_ratio = w / h
    target_width_height_ratio = target_width / target_height
    if image_width_heigt_ratio >= target_width_height_ratio:
        y1 = 0
        y2 = h - 1
        x1 = math.ceil((w - h * target_width / target_height) / 2)
        x2 = math.ceil(w - (w - h * target_width / target_height) / 2)
    else:
        x1 = 0
        x2 = w - 1
        y1 = math.ceil((h - w * target_height / target_width) / 2)
        y2 = math.ceil(h - (h - w * target_height / target_width) / 2)
    x1 = max(0, x1)
    x2 = min(x2, w - 1)
    y1 = max(0, y1)
    y2 = min(y2, h - 1)
    image = image.crop((x1, y1, x2, y2))
    image = image.resize((target_width, target_height), resample=resample)
    return image


def get_canny(
    image: np.ndarray, low_threshold: float, high_threshold: float
) -> np.ndarray:
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return image


def pad_matrix(matrix, target_shape):
    h, w, c = matrix.shape
    h1, w1 = target_shape

    if h1 < h or w1 < w:
        raise ValueError("Target shape must be larger than original shape.")

    pad_h = (h1 - h) // 2
    pad_w = (w1 - w) // 2

    padded_matrix = np.zeros((h1, w1, c))
    padded_matrix[pad_h : pad_h + h, pad_w : pad_w + w, :] = matrix

    return padded_matrix


def pad_tensor(tensor, shape):
    """
    将输入的numpy array tensor进行0填充，直到其尺寸达到目标尺寸shape。

    参数：
    tensor: numpy array，输入的tensor
    shape: tuple，目标尺寸

    返回值：
    numpy array，填充后的tensor
    """
    # 获取tensor的尺寸
    tensor_shape = tensor.shape
    # 计算需要填充的尺寸
    pad_shape = tuple(
        np.maximum(np.zeros_like(shape), np.array(shape) - np.array(tensor_shape))
    )
    # pad_shape = (np.max(0, shape[i] - tensor_shape[i]) for i in range(len(shape)))
    # 构造填充后的tensor
    pad_shape_ = ((0, x) for x in pad_shape)
    padded_tensor = np.pad(
        tensor,
        ((0, pad_shape[0]), (0, pad_shape[1]), (0, pad_shape[2]), (0, pad_shape[3])),
        # pad_shape_,
        mode="constant",
    )
    return padded_tensor


def batch_dynamic_crop_resize_images_v2(
    images: Union[torch.Tensor, np.ndarray],
    target_height: int,
    target_width: int,
    mode=Image.Resampling.LANCZOS,
) -> np.ndarray:
    """获取图像中心有效部分，并resize到对应目标宽度和高度。
        如果图像宽高比大于 target_width / target_height，则保留全部高，截取宽的中心部位；
        如果图像宽高比小于 target_width / target_height，则保留全部宽，截取高的中心部位；
        最后，将截取的图像resize到目标宽高
    Args:
        image (Image.Image): 输入图像
        target_height (int): 目标高
        target_width (int): 目标宽

    Returns:
        Image.Image: 动态截取、resize生成的图像
    """
    ndim = images.ndim
    if ndim == 4:
        b, c, h, w = images.shape
    elif ndim == 5:
        b, c, t, h, w = images.shape
        images = rearrange(images, "b c t h w->(b t) c h w")
    else:
        raise ValueError(f"ndim only support 4, 5 but given {ndim}")
    images = convert_images(
        images, data_channel_order="b c h w", return_type="pil", input_rgb_order="rgb"
    )
    images = [
        dynamic_crop_resize_image(
            image,
            target_height=target_height,
            target_width=target_width,
            resample=mode,
        )
        for image in images
    ]
    images = [to_numpy_array(x) for x in images]
    images = np.stack(images, axis=0)
    images = rearrange(images, "b h w c-> b c h w")
    if ndim == 5:
        images = rearrange(images, "(b t) c h w->b c t h w", b=b, t=t)
    return images


def batch_dynamic_crop_resize_images(
    images: Union[torch.Tensor, np.ndarray],
    target_height: int,
    target_width: int,
    mode: Literal[
        "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
    ] = "bilinear",
    # ] = "nearest",
    align_corners=False,
) -> torch.TensorType:
    """获取图像中心有效部分，并resize到对应目标宽度和高度。
        如果图像宽高比大于 target_width / target_height，则保留全部高，截取宽的中心部位；
        如果图像宽高比小于 target_width / target_height，则保留全部宽，截取高的中心部位；
        最后，将截取的图像resize到目标宽高

    Warning: 该方法对于 b c t h w t=1时 会出现图像像素错位问题，所以新增了个使用Image.Resize的V2版本
    Args:
        image (Image.Image): 输入图像
        target_height (int): 目标高
        target_width (int): 目标宽

    Returns:
        Image.Image: 动态截取、resize生成的图像
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    ndim = images.ndim

    if ndim == 4:
        b, c, h, w = images.shape
    elif ndim == 5:
        b, c, t, h, w = images.shape
        images = rearrange(images, "b c t h w->(b t) c h w")
    else:
        raise ValueError(f"ndim only support 4, 5 but given {ndim}")
    image_width_heigt_ratio = w / h
    target_width_height_ratio = target_width / target_height
    if image_width_heigt_ratio >= target_width_height_ratio:
        y1 = 0
        y2 = h - 1
        x1 = math.ceil((w - h * target_width / target_height) / 2)
        x2 = math.ceil(w - (w - h * target_width / target_height) / 2)
    else:
        x1 = 0
        x2 = w - 1
        y1 = math.ceil((h - w * target_height / target_width) / 2)
        y2 = math.ceil(h - (h - w * target_height / target_width) / 2)
    x1 = max(0, x1)
    x2 = min(x2, w - 1)
    y1 = max(0, y1)
    y2 = min(y2, h - 1)
    images = images[:, :, y1:y2, x1:x2]
    images = torch.nn.functional.interpolate(
        images,
        (target_height, target_width),
        mode=mode,  # align_corners=align_corners
    )
    if ndim == 5:
        images = rearrange(images, "(b t) c h w->b c t h w", b=b, t=t)
    return images


def his_match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src * 255.0
    dst = dst * 255.0
    src = src.astype(np.uint8)
    dst = dst.astype(np.uint8)
    res = np.zeros_like(dst)

    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))
    kw = dict(bins=256, range=(0, 256), density=True)
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side="left")
        np.clip(index, 0, 255, out=index)
        res[:, :, ch] = index[dst[:, :, ch]]
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)
    return res / 255.0
