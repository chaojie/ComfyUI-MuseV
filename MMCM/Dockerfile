# FROM mirrors.tencent.com/todacc/venus-std-base-cuda11.8:0.1.0
FROM mirrors.tencent.com/todacc/venus-std-ext-cuda11.8-pytorch2.0-tf2.12-py3.10:0.7.0

#MAINTAINER 维护者信息
LABEL MAINTAINER="anchorxia"
LABEL Email="xzqjack@hotmail.com"
LABEL Description="gpu development image, from mirrors.tencent.com/todacc/venus-std-ext-cuda11.8-pytorch2.0-tf2.12-py3.10:0.7.0"

USER root
# 安装必须软件
# RUN GENERIC_REPO_URL="http://mirrors.tencent.com/repository/generic/venus_repo/image_res" \
#     && cd /data/ \
#     && wget -q $GENERIC_REPO_URL/gcc/gcc-11.2.0.zip \
#     && unzip -q gcc-11.2.0.zip  \
#     && cd gcc-releases-gcc-11.2.0 \
#     && ./contrib/download_prerequisites \
#     && ./configure --enable-bootstrap --enable-languages=c,c++ --enable-threads=posix --enable-checking=release --enable-multilib --with-system-zlib \
#     && make --silent -j10 \
#     && make --silent install \
#     && gcc -v \
#     && rm -rf /data/gcc-releases-gcc-11.2.0 /data/gcc-11.2.0.zip 

# RUN yum update -y \
#     && yum install -y epel-release \
#     && yum install -y ffmpeg \
#     && yum install -y Xvfb \
#     && yum install -y centos-release-scl devtoolset-11
RUN yum install -y wget zsh git curl tmux cmake htop iotop git-lfs zip \
    && yum install -y autojump autojump-zsh portaudio portaudio-devel \
    && yum clean all

USER mqq
RUN source ~/.bashrc \
    && GENERIC_REPO_URL="http://mirrors.tencent.com/repository/generic/venus_repo/image_res" \
    && conda deactivate \
    # && conda remove -y -n env-2.7.18 --all \
    # && conda remove -y -n env-3.6.8 --all \
    # && conda remove -y -n env-3.7.7 --all \
    # && conda remove -y -n env-3.8.8 --all \
    # && conda remove -y -n env-3.9.2 --all \
    # && conda remove -y -n env-novelai --all \
    && conda create -n projectv python=3.10.6 -y \
    && conda activate projectv \
    && pip install venus-sdk -q -i https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple \
    --extra-index-url https://mirrors.tencent.com/pypi/simple/ \
    && pip install tensorflow==2.12.0 tensorboard==2.12.0 \
    && pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple -U \
    # 安装xformers，支持不同型号gpu
    && pip install ninja==1.11.1 \
    # && git clone https://github.com/facebookresearch/xformers.git \
    # && cd xformers \
    # && git checkout v0.0.17rc482 \
    # && git submodule update --init --recursive \
    # && pip install numpy==1.23.4 pyre-extensions==0.0.23 \
    # && FORCE_CUDA="1" MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6" pip install -e . \
    # && cd .. \
    # 安装一堆包
    && pip install --no-cache-dir transformers bitsandbytes decord accelerate xformers omegaconf einops imageio==2.31.1 \
    && pip install --no-cache-dir pandas h5py matplotlib modelcards pynvml black pytest moviepy torch-tb-profiler scikit-learn librosa ffmpeg easydict webp controlnet_aux mediapipe \
    && pip install --no-cache-dir Cython easydict gdown infomap insightface ipython librosa onnx onnxruntime onnxsim opencv_python Pillow protobuf pytube PyYAML \
    && pip install --no-cache-dir requests scipy six tqdm gradio albumentations opencv-contrib-python imageio-ffmpeg pytorch-lightning test-tube \
    && pip install --no-cache-dir timm addict yapf prettytable safetensors basicsr fvcore pycocotools wandb gunicorn \
    && pip install --no-cache-dir streamlit webdataset kornia open_clip_torch streamlit-drawable-canvas torchmetrics \
    # 安装暗水印
    && pip install --no-cache-dir invisible-watermark==0.1.5 gdown==4.5.3 ftfy==6.1.1 modelcards==0.1.6 \
    # 安装openmm相关包
    && pip install--no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.1" \
    && mim install "mmdet>=3.1.0" \
    && mim install "mmpose>=1.1.0" \
    # jupyters
    && pip install ipywidgets==8.0.3 \
    && python -m ipykernel install --user --name projectv --display-name "python(projectv)" \
    && pip install --no-cache-dir matplotlib==3.6.2 redis==4.5.1  pydantic[dotenv]==1.10.2 loguru==0.6.0 IProgress==0.4 \
    && pip install --no-cache-dir  cos-python-sdk-v5==1.9.22 coscmd==1.8.6.30 \
    # 必须放在最后pip，避免和jupyter的不兼容
    && pip install --no-cache-dir  markupsafe==2.0.1 \
    && wget -P /tmp $GENERIC_REPO_URL/cpu/clean-layer.sh \
    && sh /tmp/clean-layer.sh

ENV LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
USER root
