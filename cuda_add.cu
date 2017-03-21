/**
* @Author: Peishi Jiang <Ben1897>
* @Date:   2017-02-28T13:30:43-06:00
* @Email:  shixijps@gmail.com
* @Last modified by:   Ben1897
* @Last modified time: 2017-03-02T12:25:59-06:00
*/
/**
 * Include all the packages
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <sys/time.h>
double get_time()
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}


/**
 * [kernel GPU implementation of multivariate KDE method]
 * @param pdfset [description]
 * @param coordo [an array of the sample locations]
 * @param coordt [an array of the location of pdf to be estimated]
 * @param bd     [an array of bandwidths of the kernel]
 * @param No     [number of the given samples]
 * @param nvar   [number of variables]
 */
__global__ void kernel(int *a, int *b, int *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the index of the ith location
    // int xi;                                     // the ith location

    while (i<N) {
      c[i] = a[i] + b[i];
      i += blockDim.x*gridDim.x;
    }
}


/**
 * [cuda_kde kernel density function]
 * @param  nvar   [number of variables]
 * @param  Nt     [number of pdf to be estimated]
 * @param  No     [number of the given samples]
 * @param  bd     [an array of bandwidths of the kernel]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
int main()
{
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int N = 1e9;

  // Allocate the host memory to pdfset
  a = (int*)malloc(N*sizeof(int));
  b = (int*)malloc(N*sizeof(int));
  c = (int*)malloc(N*sizeof(int));

  // Allocate the device memory
  cudaMalloc(&d_a, N*sizeof(int));
  cudaMalloc(&d_b, N*sizeof(int));
  cudaMalloc(&d_c, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // Copy the host to the device
  cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

  // Perform cuda KDE
  double start = get_time();
  for (size_t i = 0; i < 1000; i++) {
    kernel<<<1024,1024>>>(d_a, d_b, d_c,N);
  }
  double stop;
  stop = get_time();
  printf("Time = %15.12f\n", stop - start);
  // kernel<<<grid_size,block_size>>>(d_pdfset, d_coordo, d_coordt, d_bd, d_No);

  // Copy the device back to the host
  cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n", c[N-1]);
}
