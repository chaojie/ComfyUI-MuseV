import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

IPAdapterModelDir = os.path.join(
    diffusers_path, "TMElyralab/MuseV/IP-Adapter"
)

MotionDIr = os.path.join(
    diffusers_path, "TMElyralab/MuseV/motion"
)


MODEL_CFG = {
    "IPAdapter": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "models/image_encoder"),
        "ip_ckpt": os.path.join(IPAdapterModelDir, "ip-adapter_sd15.bin"),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 4,
        "clip_embeddings_dim": 1024,
        "desp": "",
    },
    "IPAdapterPlus": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "image_encoder"),
        "ip_ckpt": os.path.join(IPAdapterModelDir, "ip-adapter-plus_sd15.bin"),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 16,
        "clip_embeddings_dim": 1024,
        "desp": "",
    },
    "IPAdapterPlus-face": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "image_encoder"),
        "ip_ckpt": os.path.join(IPAdapterModelDir, "ip-adapter-plus-face_sd15.bin"),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 16,
        "clip_embeddings_dim": 1024,
        "desp": "",
    },
    "IPAdapterFaceID": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "image_encoder"),
        "ip_ckpt": os.path.join(IPAdapterModelDir, "ip-adapter-faceid_sd15.bin"),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 4,
        "clip_embeddings_dim": 512,
        "desp": "",
    },
    "musev_referencenet": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "image_encoder"),
        "ip_ckpt": os.path.join(
            MotionDir, "musev_referencenet/ip_adapter_image_proj.bin"
        ),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 4,
        "clip_embeddings_dim": 1024,
        "desp": "",
    },
    "musev_referencenet_pose": {
        "ip_image_encoder": os.path.join(IPAdapterModelDir, "image_encoder"),
        "ip_ckpt": os.path.join(
            MotionDir, "musev_referencenet_pose/ip_adapter_image_proj.bin"
        ),
        "ip_scale": 1.0,
        "clip_extra_context_tokens": 4,
        "clip_embeddings_dim": 1024,
        "desp": "",
    },
}
