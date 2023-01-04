### Build docker image
* Build from Dockerfile
```
$ docker build -t cellular_architecture_embed:chen .
```
* Or pull from Docker Hub
```
$ docker pull pingjunchen/cellular_architecture_embed:chen
$ docker tag pingjunchen/cellular_architecture_embed:chen cellular_architecture_embed:chen
```

### Run docker container
* Start container on 172.30.205.56
```
$ docker run -it --rm --user $(id -u):$(id -g) \
  -v /rsrch1/ip/pchen6/Codes/CHEN/CellularArchitectureEmbed:/App/CellularArchitectureEmbed \
  -v /rsrch1/ip/pchen6/TMIClusterK8S/Data/CellularArchitectureEmbed:/Data \
  --shm-size=224G --gpus '"device=1,2,3"' --cpuset-cpus=10-39 \
  --name cellular_architecture_embed_chen cellular_architecture_embed:chen
```

* Start container on 10.113.120.155
```
$ docker run -it --rm --user $(id -u):$(id -g) \
  -v /rsrch1/ip/pchen6/Codes/CHEN/CellularArchitectureEmbed:/App/CellularArchitectureEmbed \
  -v /rsrch1/ip/pchen6/TMIClusterK8S/Data/CellularArchitectureEmbed:/Data \
  --shm-size=768G --gpus '"device=4,5,6,7"' --cpuset-cpus=100-255 \
  --name cellular_architecture_embed_chen cellular_architecture_embed:chen
```