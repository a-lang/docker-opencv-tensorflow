#!/bin/bash
NAME="jupyter"

docker stop $NAME
docker rm $NAME
docker ps

