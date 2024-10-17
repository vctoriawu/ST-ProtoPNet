#!/bin/sh

# change these items:
# uncomment the right DOCKER_TAG
# --name to your container's name
# --volume to the locations you have your data and repo

<< DockerTags :
DockerTags
# link to pytorch docker hub  #https://hub.docker.com/r/pytorch/pytorch/tags
#DOCKER_TAG=pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
#DOCKER_TAG=torch_hhprotonet

# DGX and Purang30
# DOCKER_TAG=pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
DOCKER_TAG=torch2.3.0-cuda11.8

<< DockerContainerBuild :
DockerContainerBuild
docker run -it --ipc=host \
      --gpus device=ALL \
      --name=Victoria_Torch_STProtoNet  \
      --volume=$HOME/workspace/ST-ProtoPNet:/workspace/ST-ProtoPNet \
      --volume=/raid/home/hoomanv/workspace/Datasets/CUB_200_2011:/workspace/ST-ProtoPNet/data/CUB_200_2011 \
      $DOCKER_TAG
