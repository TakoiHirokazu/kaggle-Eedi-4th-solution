FROM nvcr.io/nvidia/pytorch:23.11-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    tmux &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install iterative-stratification==0.1.7 torch-ema==0.3 transformers==4.44.2 lightgbm==4.2.0 \
    catboost==1.2.2 optuna==3.5.0 polars==0.20.18 gensim==4.3.2 sentencepiece==0.1.99 holidays==0.40 \
    albumentations==1.1.0 timm==0.9.12 opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 h5py==3.11.0\
    datasets==2.19.2 huggingface-hub==0.23.2 peft==0.12.0 bitsandbytes==0.43.1 accelerate==0.32.1 trl==0.11.4\
    sentence_transformers==3.1.0 optimum auto-gptq

# torchaudio.
RUN git clone https://github.com/pytorch/audio && \
    cd audio && \
    git checkout tags/v2.0.1 && \
    python setup.py develop && \
    cd ..

WORKDIR /tmp/working