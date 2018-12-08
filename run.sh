#!/bin/bash
LOCAL_DIR="$PWD/notebook"
IMAGE="alangtw/jupyter-opencv-tf:181206"
NAME="jupyter"

docker run -d --name=$NAME \
 -p 8888:8888 \
 -v $LOCAL_DIR:/home/jovyan/work \
 $IMAGE

docker ps

