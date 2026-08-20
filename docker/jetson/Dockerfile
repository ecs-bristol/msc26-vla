FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PYTHONPATH=/workspace/project/src

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        "numpy>=2.0,<2.3" \
        "pydantic>=2.11,<3" \
        "PyYAML>=6.0,<7" \
        "Pillow>=11,<13" \
        "requests>=2.32,<3" \
        "psutil>=7,<8" \
        "h5py>=3.12,<4" \
        "pytest>=8.3,<9" \
        "lerobot[smolvla]>=0.6.1,<0.7.0"

# The NVIDIA PyTorch image needs a NumPy 1.x runtime bridge on Jetson.
RUN python3 -m pip install --no-cache-dir --force-reinstall \
        "numpy>=1.26,<2"

WORKDIR /workspace/project
CMD ["bash"]
