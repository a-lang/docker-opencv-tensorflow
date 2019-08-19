Jupyter notebook docker image with Python, OpenCV and Tensorflow for Deep Learning development.

### What packages were included in the image

![version_info](version_info.png)

### Run the pre-built image

Go to the [Docker Hub](https://hub.docker.com/r/alangtw/jupyter-opencv-tf/)

```bash
git clone https://github.com/a-lang/docker-opencv-tensorflow.git

docker run -it --name=jupyter \
-p 8888:8888 \
-v $PWD/notebook:/home/jovyan/work \
alangtw/jupyter-opencv-tf:181206
```

alternatively,

```bash
./run.sh
```

Change the permission of the directory notebook otherwise the notebook will be locked as read-only.

```
chown -R 1000 $PWD/notebook
```

### Get the token info of Jupyter Notebook web

```bash
docker exec -it jupyter jupyter notebook list
```

### Build the image (optional)

```bash
docker build -t jupyter-opencv-tf .
```

### Add other python modules (optional)

Install cmake, dlib modules with pip

```
docker exec -it jupyter bash

jovyan@176594c4a5e5:~$ pip install cmake
jovyan@176594c4a5e5:~$ pip install dlib
```

Save the container changed into a new image

```
docker commit jupyter jupyter-opencv-tf:new
```

