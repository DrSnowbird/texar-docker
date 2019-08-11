ARG CUDA_VER=${CUDA:-10.1}
FROM nvidia/cuda:${CUDA_VER}-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         curl \
         ca-certificates \
     && rm -rf /var/lib/apt/lists/*

#### ---- Conda setup ----
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION  && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

#### ---- Texar setup ----
WORKDIR /opt/texar
COPY . .

RUN echo "current directory: `pwd`" && ls -al .

#### ---- tensorflow setup ----
RUN pip install -e . \
    && pip install -r requirements.txt

ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA_VER}/compat:$LD_LIBRARY_PATH

RUN pip list | tee installed_pip_modules.txt

RUN apt-get update

WORKDIR /workspace
RUN chmod -R a+w /workspace

