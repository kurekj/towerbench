# Pinned environment of the towerbench sweep (PyTorch 2.11, CUDA 12.8, Python 3.11).
#   docker build -t towerbench .
#   docker run --gpus all -it -v $HOME/towerbench:/root/towerbench towerbench towerbench run ml-1m ease
# Data, results and logs live under /root/towerbench (TOWERBENCH_ROOT); mount it to keep them.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
ENV TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/towerbench
COPY requirements-lock.txt ./
RUN grep -v '^torch' requirements-lock.txt > /tmp/req.txt && pip install -r /tmp/req.txt
COPY . .
RUN pip install --no-deps -e .
ENV TOWERBENCH_ROOT=/root/towerbench
CMD ["towerbench", "--help"]
