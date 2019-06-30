# Usage Example
We expect that the user loads the docker image on Linux. The following instructions are based on Ubuntu 16.04.1 LTS  
## 1. Load Docker Image

Load image from docker hub

```
	sudo docker pull dgl2019/icse2019-artifacts:latest
```

If your system doesn't have the "docker" command. please install it as follows:
```
	sudo apt-get install docker.io
```

If the docker image is loaded successfully, then run:

```
 sudo docker images
``` 
and you will find a repository named "dgl2019/icse2019-artifacts" in your list as follows:

```
REPOSITORY                    TAG                 IMAGE ID            CREATED             SIZE
dgl2019/icse2019-artifacts    latest              7f57cfe48ab8        About an hour ago   6.43GB

```

## 2. Run Docker Image

#### NOTE: 7f57cfe48ab8 is the  IMAGE ID which can be found in the last step.
```
sudo docker run -it 7f57cfe48ab8 bash
```
or 

```
sudo docker run -it dgl2019/icse2019-artifacts:latest  bash
```

If everything goes well, you will enter a new bash shell like:

```
root@09f4131f7471:/#
``` 







