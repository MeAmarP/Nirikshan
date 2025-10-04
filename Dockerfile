FROM ubuntu:22.04

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

WORKDIR /workspace

# Copy dependency definition and create the conda environment
COPY myenv.yml ./myenv.yml
RUN conda env create -f myenv.yml && conda clean -afy

# Activate the project environment by default for subsequent commands and runtime
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH=/opt/conda/envs/myenv/bin:/opt/conda/bin:$PATH


# Copy project code into the image
COPY . .

# Default entrypoint runs the analytics pipeline; override CMD to supply args
ENTRYPOINT ["python", "src/main.py"]
# CMD ["--fpath", "/workspace/data/sample.mp4"]

# Instructions to run container with GPU and display support
# Use the following command to run the container:
# docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --network host <image_name>
