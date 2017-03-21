/**
* @Author: Peishi Jiang <Ben1897>
* @Date:   2017-02-27T10:13:51-06:00
* @Email:  shixijps@gmail.com
* @Last modified by:   Ben1897
* @Last modified time: 2017-03-01T14:37:43-06:00
*/

/**
 * Include all the packages
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void kernel(double *pdfset, double *coordo,
                       double *coordt, double bd, int No)
{
    double pdf = 0.;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int xi = coordt[i];

    for (int j = 0; j < No; j++) {
      double u = (xi - coordo[j]) / bd;
      if (u*u < 1) {
        double kernel = 0.75 * (1-u*u);  // the Epanechnikov kernel
        pdf = pdf + kernel/bd;
      }
    }

    pdfset[i] = pdf/(double)No;
}


/**
 * [cuda_kde 1D kernel density function]
 * @param  bd     [bandwidth of the kernel]
 * @param  Nt     [number of pdf to be estimated]
 * @param  No     [number of the given samples]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
extern "C" {
double *cuda_kde(double bd, int Nt, int No, double *coordo, double *coordt)
{
  int    blockWidth = 1024;
  int    blocksX = Nt/blockWidth+1;
  double *pdfset, *d_pdfset;    // The pdf array to be estimated
  double *d_coordo, *d_coordt;
  // double d_bd, d_No;

  // Allocate the host memory to pdfset
  pdfset = (double*)malloc(Nt*sizeof(double));
  //   datat = new double(2*N);

  // Allocate the device memory
  cudaMalloc(&d_pdfset, Nt*sizeof(double));
  cudaMalloc(&d_coordt, Nt*sizeof(double));
  cudaMalloc(&d_coordo, No*sizeof(double));

  // Copy the host to the device
  cudaMemcpy(d_coordt, coordt, Nt*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coordo, coordo, No*sizeof(double), cudaMemcpyHostToDevice);

  // Set d_pdfset to zero
  cudaMemset(d_pdfset, 0, Nt*sizeof(double));

  // Perform cuda KDE
  const dim3 block_size(blockWidth, 1, 1);
  const dim3 grid_size(blocksX, 1, 1);
  kernel<<<grid_size,block_size>>>(d_pdfset, d_coordo, d_coordt, bd, No);
  // kernel<<<grid_size,block_size>>>(d_pdfset, d_coordo, d_coordt, d_bd, d_No);

  // Copy the device back to the host
  cudaMemcpy(pdfset, d_pdfset, Nt*sizeof(double), cudaMemcpyDeviceToHost);

  // Free the memory
  cudaFree(d_coordo);
  cudaFree(d_coordt);
  cudaFree(d_pdfset);

  return pdfset;
}
}
