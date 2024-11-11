docker run -it --gpus all --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/workspace -v ~/Documents/workspace/models:/models --name mydev nirikshan-dev:v1

