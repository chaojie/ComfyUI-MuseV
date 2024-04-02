import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MotionDIr = os.path.join(
    diffusers_path, "TMElyralab/MuseV/motion"
)

MODEL_CFG = {
    "musev_referencenet": {
        "net": os.path.join(MotionDIr, "musev_referencenet"),
        "desp": "",
    },
}
