# LungEvolutionPathomics
### Pipeline of investigating lung preneoplasia evolution via pathomics
* Data to reproduce the results was deposited in [Synapse:syn48334123](https://www.synapse.org/#!Synapse:syn48334123)
* [NucleiSegHE](https://github.com/cpathology/NucleiSegHE) was used to generate ROI-level nuclei segmentation 
![Pipeline of Lung Evolution Pathomics](LungEvolutionPathomics.png)


## Environment Configurations
### a. Prepare docker image
* Build from Dockerfile
```
$ docker build -t lung_evolution_pathomics:chen .
```
* Or pull from Docker Hub
```
$ docker pull pingjunchen/lung_evolution_pathomics:chen
$ docker tag pingjunchen/lung_evolution_pathomics:chen lung_evolution_pathomics:chen
```

### b. Setup docker container
* Start docker container (specify CODE_ROOT & DATA_ROOT)
```
$ docker run -it --rm --user $(id -u):$(id -g) \
  -v ${CODE_ROOT}:/App/LungEvolutionPathomics \
  -v ${DATA_ROOT}:/Data \
  --shm-size=224G --gpus '"device=0,1,2,3"' --cpuset-cpus=0-39 \
  --name lung_evolution_pathomics_chen lung_evolution_pathomics:chen
```

## Pathomic Features Evaluating Lung Preneoplasia Evolution
### 1. Evolution trends along with pathological stage progression
### 2. Correlation with a series of molecular markers