FROM nvcr.io/nvidia/pytorch:23.10-py3

# Upgrade pip
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install git -y

RUN mkdir -p /workspace 
WORKDIR /workspace 

EXPOSE 8000:8000
