import os
from typing import List, Literal, Tuple, Union
import cv2
from einops import rearrange, repeat

import numpy as np
from PIL import Image
import torch


def convert_images(
    data: Union[
        str, List[str], Image.Image, List[Image.Image], np.ndarray, torch.Tensor
    ],
    return_type: str = "numpy",
    data_channel_order: str = "b h w c",
    input_rgb_order: str = "rgb",
    return_rgb_order: str = "rgb",
    return_data_channel_order: str = "b h w c",
) -> Union[np.ndarray, List[Image.Image], torch.Tensor]:
    """将所有图像数据都先转换成numpy b*c*h*w格式，再根据return_type转换成目标格式。
    Args:
        data (Union[str, List[str], Image.Image, List[Image.Image], np.ndarray]): _description_
        return_type (str, optional): 返回的图像格式. Defaults to "numpy". 候选项
            numpy
            torch
            pil
            opencv
        rgb_order (str, optional): 输入图像的通道格式, 默认是"rgb" 格式，候选项
            rgb
            bgr
    Raises:
        ValueError: only support return_type (numpy, torch, pil), but given return_type

    Returns:
        Union[np.ndarray, List[Image.Image], torch.Tensor]: _description_
    """

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if isinstance(data, (str, Image.Image)):
        data = [data]
    if isinstance(data, list):
        if isinstance(data[0], str):
            data = [Image.open(image) for image in data]
            if data_channel_order == "rgb":
                data = [image.convert("RGB") for image in data]
        if isinstance(data[0], Image.Image):
            data = [np.asarray(image) for image in data]
        if isinstance(data[0], np.ndarray):
            data = np.stack(data)

    if isinstance(data, np.ndarray):
        if data.ndim == 5:
            data = rearrange(data, "{}-> (b t) h w c".format(data_channel_order))
        elif data.ndim == 4:
            if data_channel_order != "b h w c":
                data = rearrange(data, "{}-> b h w c".format(data_channel_order))
        elif data.ndim == 3:
            if data_channel_order != "h w c":
                data = rearrange(data, "{}-> h w c".format(data_channel_order))
            data = data[np.newaxis, ...]
    if input_rgb_order != return_rgb_order:
        data = data[..., ::-1]
    if return_data_channel_order != "b h w c":
        data = rearrange(data, "b h w c -> {}".format(return_data_channel_order))
    c_idx = return_data_channel_order.split(" ").index("c")
    if return_type == "numpy":
        return data
    elif return_type == "torch":
        return torch.from_numpy(data)
    elif return_type.lower() == "pil":
        data = data.astype(np.uint8)
        data = [Image.fromarray(data[i]) for i in range(len(data))]
        return data
    elif return_type == "opencv":
        data = data.transpose(0, 2, 3, 1)
        return data
    else:
        raise ValueError(
            f"only support return_type (numpy, torch, PIL), but given {return_type}"
        )


def is_video(path: str, exts=["mp4", "mkv", "ts", "rmvb", "mov", "avi"]):
    path_ext = os.path.splitext(os.path.basename(path))[-1][1:].lower()
    return path_ext in exts


def pil_read_image_with_alpha(
    img_path: str,
    color_channel: str = "rgb",
    return_type: Literal["numpy", "PIL"] = "PIL",
) -> Tuple[Image.Image, np.ndarray]:
    image_with_alpha = Image.open(img_path)
    if image_with_alpha.mode != "RGBA":
        image_with_alpha = image_with_alpha.convert("RGBA")
    background = Image.new("RGB", image_with_alpha.size, (255, 255, 255))
    if background.mode != "RGBA":
        background = background.convert("RGBA")
    rgb_image = Image.alpha_composite(background, image_with_alpha)
    if color_channel == "rgb":
        rgb_image = rgb_image.convert("RGB")
    if return_type == "numpy":
        rgb_image = np.array(rgb_image)
    return rgb_image


# 该部分代码不是很正常，可以使用PIL.Image.Image
def opencv_cvt_alpha(image: np.ndarray) -> np.ndarray:
    """read image with alpha channel, and fill alpha channel with white background.

    Args:
        img (str): opencv read image with alpha channel
    Returns:
        np.ndarray: image array
    """
    # make mask of where the transparent bits are
    trans_mask = image[:, :, 3] == 0

    # replace areas of transparency with white and not transparent
    image[trans_mask] = [255, 255, 255, 255]

    # new image without alpha channel...
    new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return new_image


def read_image_with_alpha(path: str, color_channel: str = "rgb") -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = opencv_cvt_alpha(img)
    if color_channel == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_image_as_5d(path: str, color_channel: str = "rgb") -> np.ndarray:
    img = pil_read_image_with_alpha(path, color_channel, return_type="numpy")
    img = repeat(img, "h w c-> b c t h w", b=1, t=1)
    return img


def is_image(path: str, exts=["jpg", "png", "jpeg", "webp"]):
    path_ext = os.path.splitext(os.path.basename(path))[-1][1:].lower()
    return path_ext in exts
