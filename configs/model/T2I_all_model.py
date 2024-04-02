import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

T2IDir = os.path.join(
    diffusers_path, "TMElyralab/MuseV/t2i"
)

MODEL_CFG = {
    "majicmixRealv6Fp16": {
        "sd": os.path.join(T2IDir, "sd1.5/majicmixRealv6Fp16"),
    },
    "fantasticmix_v10": {
        "sd": os.path.join(T2IDir, "sd1.5/fantasticmix_v10"),
    },
}
