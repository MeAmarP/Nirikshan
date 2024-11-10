FROM ubuntu:22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

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
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    chmod +x /miniconda.sh && \
    /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# /home/mighty/Documents/workspace/Nirikshan/myenv.yml
# Create and activate the conda environment using the provided yml file
COPY myenv.yml /workspace/myenv.yml
RUN conda env create -f /workspace/myenv.yml 

# Set up environment to activate conda environment by default
# SHELL ["/bin/bash", "-c"]
# RUN echo "conda activate myenv" >> ~/.bashrc


# Set working directory
WORKDIR /workspace


# Default command
# CMD ["/bin/bash"]

# Instructions to run container with GPU and display support
# Use the following command to run the container:
# docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --network host <image_name>

