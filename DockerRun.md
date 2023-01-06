### Build docker image
* Build from Dockerfile
```
$ docker build -t cellular_architecture:chen .
```
* Or pull from Docker Hub
```
$ docker pull pingjunchen/cellular_architecture:chen
$ docker tag pingjunchen/cellular_architecture:chen cellular_architecture:chen
```

### Run docker container

* Start container on 10.113.120.155
```
$ docker run -it --rm --user $(id -u):$(id -g) \
  -v /rsrch1/ip/pchen6/Codes/CHEN/CellularArchitecture:/App/CellularArchitecture \
  -v /rsrch1/ip/pchen6/CellularArchitectureData:/Data \
  --shm-size=768G --gpus '"device=2,3,4,5,6,7"' --cpuset-cpus=100-255 \
  --name cellular_architecture_chen cellular_architecture:chen
```