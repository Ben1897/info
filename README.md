# An Information-Theoretic Framework for Complex System Analysis
================================================================

## Introduction
This is an open-source program for evaluating the dynamics of complex system by using an information-theoretic framework based on time series observations.

## References
- Jiang, P., & Kumar, P. (2018). Interactions of information transfer along separable causal paths. Physical Review E, 97(4), 042310.
- Jiang, P., & Kumar, P. (2018). Information transfer from causal history in complex system dynamics. Physical Review E, in review.

## Requirements
1. Add the folder into your environment variable PYTHONPATH.
2. Install required python packages using *pip*: `pip install -r requirements.txt`.
3. Generate the dynamic libraries for the Kernal Density Estimation (KDE) GPU by:
```
cd [info_folder]/info/info/utils/
make clean; make
```
5. Generate the dynamic libraries for the K-Nearest Neighbor (KNN) GPU by (Note that it is suggested to use scipy's own KNN approach):
    - Download the [Fast K-Nearest Neighbor search with GPU](https://github.com/PeishiJiang/knn_cuda) (Note that in Peishi's version the distance calculation is based on the maximum norm).
    - Specify `PYTHON_INCLUDE`, `PYTHON_LIB`, `CUDA_DIR` correctly in the `Makefile.config`.
    - Install [Boost](http://www.boost.org/) to enable the connection between C++ and Python code.
    - Type `make` to generate the `knn.so` dynamic library.
    - Copy `knn.so` to the utility folder by typing:
```
cp [knn-folder]/knn.so [info-folder]/info/info/utils/
```
6. [TODO] make file for pdf estimation codes

## License
This software is freeware and is released under GNU GPL. See LICENSE file for more information.

## Contacts
Peishi Jiang (pjiang6@illinois.edu), Praveen Kumar (kumar1@illinois.edu)
