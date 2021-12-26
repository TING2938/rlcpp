FROM ubuntu:20.04

WORKDIR /work

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get install -y vim git cmake make build-essential wget unzip \
    python3 python3-dev python3-pip && \ 
    apt clean && \
    pip3 install gym

RUN git clone https://github.com/clab/dynet.git && \
    cd dynet && \
    mkdir eigen && \
    cd eigen && \
    wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip && \
    unzip eigen-b2e267dc99d4.zip && \
    cd .. && mkdir build && cd build && \
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen && \
    make -j 4 && \
    make install && \
    cd ../../ && rm -rf dynet 

RUN echo 'export PYTHONPATH=/work/rlcpp/env/gym_cpp/gym_wrapper:$PYTHONPATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

RUN git clone https://gitlab.com/TING2938/rlcpp.git && \
    cd rlcpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j 4

###############################################################################################
CMD ["bash", "-c", "/work/rlcpp/build/train/train_dqn_dynet"]





