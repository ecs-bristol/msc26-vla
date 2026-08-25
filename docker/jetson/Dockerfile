FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PYTHONPATH=/workspace/project/src

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        "numpy>=2.0,<2.3" \
        "opencv-python-headless>=4.9,<4.14" \
        "pydantic>=2.11,<3" \
        "PyYAML>=6.0,<7" \
        "Pillow>=11,<13" \
        "einops>=0.8,<0.9" \
        "draccus>=0.11.6,<0.12" \
        "huggingface-hub>=1.6,<2" \
        "requests>=2.32,<3" \
        "gymnasium>=1.1.1,<2" \
        "safetensors>=0.4.3,<1" \
        "packaging>=24.2,<26" \
        "termcolor>=2.4,<4" \
        "tqdm>=4.66,<5" \
        "cmake>=3.29.0.1,<4.2" \
        "setuptools>=71,<82" \
        "transformers>=5.4,<5.6" \
        "num2words>=0.5.14,<0.6" \
        "accelerate>=1.14,<2" \
        "psutil>=7,<8" \
        "h5py>=3.12,<4" \
        "pytest>=8.3,<9" && \
    python3 -m pip install --no-cache-dir --no-deps "lerobot==0.6.1"

# The NVIDIA PyTorch image needs a NumPy 1.x runtime bridge on Jetson.
RUN python3 -m pip install --no-cache-dir --force-reinstall \
        "numpy>=1.26,<2"

# LeRobot's stable torchvision lower bound would otherwise replace NVIDIA's
# Jetson-specific pre-release torch/torchvision pair with generic CUDA wheels.
RUN python3 -c 'import torch, torchvision; assert ".nv25." in torch.__version__, torch.__version__; assert torchvision.__version__.startswith("0.22.0a0+"), torchvision.__version__'

WORKDIR /workspace/project
CMD ["bash"]
