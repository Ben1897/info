# The make file is used to generate the shared library of the c code and
# the cuda code for calculating KDE
#
# @Author: Peishi Jiang <Ben1897>
# @Date:   2017-02-27T16:56:56-06:00
# @Email:  shixijps@gmail.com
# @Last modified by:   Ben1897
# @Last modified time: 2017-03-01T11:57:57-06:00

all: kde.so kde1d.so cuda_kde.so cuda_kde1d.so

# Generate the share library from kde.c
kde.so: kde.o
	gcc -shared -o kde.so kde.o
kde.o: kde.c
	gcc -c -Wall -Werror -fpic kde.c -std=c99
# Generate the share library from kde1d.c
kde1d.so: kde1d.o
	gcc -shared -o kde1d.so kde1d.o
kde1d.o: kde1d.c
	gcc -c -Wall -Werror -fpic kde1d.c -std=c99
# Generate the share library from cuda_kde.cu
cuda_kde.so: cuda_kde.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda_kde.so --shared cuda_kde.cu
# Generate the share library from cuda_kde1d.cu
cuda_kde1d.so: cuda_kde1d.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda_kde1d.so --shared cuda_kde1d.cu

clean:
	rm *.o *.so
# gcc -c -Wall -Werror -fpic kde.c -std=c99
# gcc -shared -o kde.so kde.o
# nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda_kde.so --shared cuda_kde.cu
