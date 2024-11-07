# Use NVIDIA's base image with CUDA, cuDNN, and Python pre-installed
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables to accept prompts and use Python 3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    chmod +x /miniconda.sh && \
    /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create and activate the conda environment using the provided yml file
COPY myenv.yml /workspace/myenv.yml
RUN conda env create -f /workspace/myenv.yml && conda clean -a

# Set up environment to activate conda environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate dev-env" >> ~/.bashrc

# Install additional dependencies for popular deep learning frameworks
RUN conda activate dev-env && \
    conda install -c conda-forge cudatoolkit=11.8 cudnn && \
    conda install -c conda-forge pytorch torchvision torchaudio && \
    pip install tensorflow paddlepaddle && \
    pip install transformers

# Set working directory
WORKDIR /workspace

# Set up virtual display for GUI applications
RUN apt-get update && apt-get install -y \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Default command
CMD ["/bin/bash"]

# Instructions to run container with GPU and display support
# Use the following command to run the container:
# docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --network host <image_name>

