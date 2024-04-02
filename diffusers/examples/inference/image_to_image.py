import warnings

from MuseVdiffusers import StableDiffusionImg2ImgPipeline  # noqa F401


warnings.warn(
    "The `image_to_image.py` script is outdated. Please use directly `from MuseVdiffusers import"
    " StableDiffusionImg2ImgPipeline` instead."
)
