# base image with cuda 12.1
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y wget curl git vim sudo cmake build-essential \
    libssl-dev libffi-dev python3-dev python3-venv python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip3 install packaging psutil pexpect ipywidgets jupyterlab ipykernel \
    librosa soundfile

# upgrade pip
RUN pip3 install --upgrade pip

# install pruna (v0.2.7 is not published on PyPI; install from Git tag)
# use --no-deps because core deps are installed separately in this image
RUN pip3 install --no-deps git+https://github.com/PrunaAI/pruna.git@v0.2.7

# install ipython kernel
RUN python3 -m ipykernel install --user --name pruna_cuda12 --display-name "Python (pruna_cuda12)"

# install torch with cuda support
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu121

# install remaining dependencies from PyPI
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# optionally pre-download weights at build time
# default is off to avoid build failures due to transient HF/network/auth issues
ARG PRELOAD_WEIGHTS=0
RUN if [ "$PRELOAD_WEIGHTS" = "1" ]; then \
      python3 /download_weights.py; \
    else \
      echo "Skipping model pre-download at build time (PRELOAD_WEIGHTS=$PRELOAD_WEIGHTS)"; \
    fi

# run the handler
CMD python3 -u /handler.py
