FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN git clone https://github.com/renning22/FastChat.git
RUN cd FastChat && pip3 install -e ".[model_worker,webui]"