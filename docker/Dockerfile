FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y python3.11 python3.11-distutils curl git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.11 get-pip.py

RUN git clone https://github.com/renning22/FastChat.git
RUN cd FastChat && pip3.11 install -e ".[model_worker,webui]"

RUN pip3.11 install vllm