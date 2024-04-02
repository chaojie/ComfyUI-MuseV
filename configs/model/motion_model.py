import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MotionDIr = os.path.join(
    diffusers_path, "TMElyralab/MuseV/motion"
)


MODEL_CFG = {
    "musev": {
        "unet": os.path.join(MotionDIr, "musev"),
        "desp": "only train unet motion module, fix t2i",
    },
    "musev_referencenet": {
        "unet": os.path.join(MotionDIr, "musev_referencenet"),
        "desp": "train referencenet, IPAdapter and unet motion module, fix t2i",
    },
    "musev_referencenet_pose": {
        "unet": os.path.join(MotionDIr, "musev_referencenet_pose"),
        "desp": "train  unet motion module and IPAdapter, fix t2i and referencenet",
    },
}
