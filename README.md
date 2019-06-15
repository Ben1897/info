# An Information-Theoretic Framework for Causal History Analysis in Complex System Dynamics
===========================================================================================

## Introduction
This is an open-source program for evaluating the complex system dynamics by using a recently-proposed causal history analysis framework.

![Illustration of Causal History Analysis Framework in a Quadvariate system](FigDiagram.jpg = 100x)

## References
- Jiang, P., & Kumar, P. (2019). Bundled interaction from causal history in complex system dynamics. submitted.
- Jiang, P., & Kumar, P. (2019). Using information flow for whole system understanding from component dynamics. submitted.
- Jiang, P., & Kumar, P. (2019). Information transfer from causal history in complex system dynamics. Physical Review E, 99(1), 012306.
- Jiang, P., & Kumar, P. (2018). Interactions of information transfer along separable causal paths. Physical Review E, 97(4), 042310.

## Code Structure
```
|   README.md
|   environment.yml
|
|---info                         # the main code folder
|   |   core                     # the core python files
|   |   |   info.py              # the code for computing information measures
|   |   |   info_network.py      # the code for computing information measures in a DAG
|   |   models                   # several synthetic models
|   |   |   logistic.py          # the multivariate noisy logistic model
|   |   |   others.py            # other models
|   |   utils                    # utility functions used by codes in the core and models folders
|
|---examples                     # the example folders (including jupyter notebook for illustration)
|   |   logistic                 # causal history analysis for logistic model
|   |   OU                       # causal history analysis for Ornstein–Uhlenbeck process
|   |   Lorenz                   # causal history analysis for the Lorenz model
|   |   streamchemistry          # causal history analysis for an observed stream chemistry dynamics
|   |   bundled_streamchemistry  # bundled causal history analysis for an observed stream chemistry dynamics
```

## Requirements
1. Add the folder into your environment variable PYTHONPATH.
2. Create the environment from the `environment.yml` file by:
```
conda env create -f environment.yml
```
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

## License
This software is freeware and is released under GNU GPL. See LICENSE file for more information.

## Contacts
Peishi Jiang (pjiang6@illinois.edu), Praveen Kumar (kumar1@illinois.edu)
