FROM nvidia/cuda:11.6.0-devel-ubuntu18.04 as gpu

WORKDIR /usr/src

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get upgrade -y
RUN apt-get install -y tzdata
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    git

ENV PYTHON_VERSION 3.7.12
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT &&  \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT

RUN pip3 install --upgrade pip \
    && pip3 install poetry \
    && poetry config virtualenvs.create false

COPY pyproject.toml /usr/src
RUN poetry install --no-root

RUN pip3 install dlib

RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY ./ /usr/src
