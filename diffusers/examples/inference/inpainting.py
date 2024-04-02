import warnings

from MuseVdiffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline  # noqa F401


warnings.warn(
    "The `inpainting.py` script is outdated. Please use directly `from MuseVdiffusers import"
    " StableDiffusionInpaintPipeline` instead."
)
