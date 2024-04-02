from easydict import EasyDict as edct

ModelCfg = edct()

# a full path of models list  should be
# ModelCfg.StableDiffusion = {
#         "model_path": str,
#         "vae_path": str,
# }

# the full path is os.path.join(ModelCfg.Online_Model_Dir, ModelCfg.model_name["k"])
# Online_Model_Dir use cos

ModelCfg.Online_Model_Dir = "/DeployModels/vision"
ModelCfg.Offline_Model_Dir = "/cfs/cfs-4a8cd28be/DeployModel/vision"

# transition model
ModelCfg.TransnetV2 = {"model_path": "transition/transnetv2_pytorch_weights.pth"}

# generation model path
Generation = edct()
## sd model_path
StableDiffusion = edct()
Generation.StableDiffusionModel = StableDiffusion
ModelCfg.Generation = Generation
StableDiffusion.StableDiffusion = {
    "model_path": "sd-v1-4/snapshots/2881c082ee0dc70d9eeb645f1b150040a4b62767"
}
StableDiffusion.WaifuSD_V13_FP32 = {
    "model_path": "waifu-diffusion-v1-3_FP32",
}
StableDiffusion.WaifuSD_V13_FP16 = {
    "model_path": "waifu-diffusion-v1-3_FP16",
}
StableDiffusion.WaifuSD = StableDiffusion.WaifuSD_V13_FP32
StableDiffusion.AnythingSD = {
    "model_path": "anything-v3.0",
}
StableDiffusion.NovelAISDInformal = {
    "model_path": "NovelAISDInformal",
}
StableDiffusion.TrinartSD = {
    "model_path": "models--naclbit--trinart_stable_diffusion_v2/snapshots/0dcafd78d07345d30f3e7c12277693e0ffeeec72",
}
StableDiffusion.ArcherSD = {
    "model_path": "archer-diffusion",
}
ModelCfg.TrinartSD = {
    "model_path": "sd-v1-4/snapshots/2881c082ee0dc70d9eeb645f1b150040a4b62767",
}
StableDiffusion.PixelSD = {
    "model_path": "All-In-One-Pixel-Model",
}
StableDiffusion.PixelArtSD = {
    "model_path": "PixelArtSD",
}
StableDiffusion.MomokoE = {
    "model_path": "momoko-e",
}
