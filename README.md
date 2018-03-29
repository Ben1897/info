# An Application of Information Theory in Ecohydrology
======================================================

## Introduction
[TODO]

## References
[TODO]

## Requirements
1. Add the folder into your environment variable PYTHONPATH.
2. Install required python packages using *pip*: `pip install -r requirements.txt`.
3. Generate the dynamic libraries for the Kernal Density Estimation (KDE) GPU by:
```
cd [info_folder]/info/info/utils/
make clean; make
```
5. Generate the dynamic libraries for the K-Nearest Neighbor (KNN) GPU by:
...* Download the [Fast K-Nearest Neighbor search with GPU](https://github.com/chrischoy/knn_cuda).
...* Specify `PYTHON_INCLUDE`, `PYTHON_LIB`, `CUDA_DIR` correctly in the `Makefile.config`.
...* Install [Boost](http://www.boost.org/) to enable the connection between C++ and Python code.
...* Type `make` to generate the `knn.so` dynamic library.
...* Copy `knn.so` to the utility folder by typing:
```
cp [knn-folder]/knn.so [info-folder]/info/info/utils/
```
6. [TODO] make file for pdf estimation codes

## Contacts
Peishi Jiang (pjiang6@illinois.edu), Praveen Kumar (kumar1@illinois.edu)
