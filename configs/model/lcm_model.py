import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

LCMDir = os.path.join(
    diffusers_path, "TMElyralab/MuseV/lcm"
)


MODEL_CFG = {
    "lcm": {
        os.path.join(LCMDir, "lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"): {
            "strength": 1.0,
            "lora_block_weight": "ALL",
            "strength_offset": 0,
        },
    },
}
