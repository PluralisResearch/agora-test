FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  build-essential \
  rsync \
  openssh-client \
  curl \
  wget \
  git \
  vim \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install python
RUN conda install python=3.13 "pip>=25.3"

# Install uv
RUN pip install uv

# Install torch with CUDA 12.8 support
RUN uv pip install --system --no-cache torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Use this to force dependencies in case of conflicts etc.
COPY constraints.txt constraints.txt

# Install Pithos and agora_server from the staged checkout
COPY pithos/ pithos/
RUN cd pithos && uv pip install --system --build-constraint ../constraints.txt -e .

COPY agora_server/ agora_server/
RUN cd agora_server && uv pip install --system -e .

# Install agora
COPY agora/ agora/
RUN cd agora && uv pip install --system --build-constraint ../constraints.txt -e .

CMD ["bash"]
