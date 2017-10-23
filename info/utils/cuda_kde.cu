/**
* @Author: Peishi Jiang <Ben1897>
* @Date:   2017-02-28T13:30:43-06:00
* @Email:  shixijps@gmail.com
* @Last modified by:   Ben1897
* @Last modified time: 2017-03-02T12:33:56-06:00
*/
/**
 * Include all the packages
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

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
 * @param Nt     [number of pdf to be estimated]
 * @param nvar   [number of variables]
 */

// Gaussian kernel
__global__ void gaussian_kernel(double *pdfset, double *coordo,
                       double *coordt, double *bd,
                       int No, int Nt, int nvar)
{
    double pdf;                               // the pdf of the ith location
    double prod_kern;                              // the kernel information of the ith location
    double u, kernel;                              // the kernel information of the ith location in the kth variable
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the index of the ith location
    // int xi;                                        // the ith location

    /* if (i >= blockDim.x*gridDim.x) { */
    /*   return; */
    /* } */

    while (i < Nt) {
      pdf = 0.;
      for (int j = 0; j < No; j++)
      {
        prod_kern = 1.;
        for (int k = 0; k < nvar; k++)
        {
          u  = (coordt[i*nvar+k] - coordo[j*nvar+k]) / bd[k];

          kernel = 1./sqrt(2*M_PI) * exp(-1./2.*pow(u,2));
          prod_kern = prod_kern * kernel/bd[k];
        }
        // if (prod_kern > 0) printf("prod_kern %f\n", prod_kern);
        pdf = pdf + prod_kern;
      }

      pdfset[i] = pdf/(double)No;

      // Roll the block to see whether (for 1D)
      i += blockDim.x*gridDim.x;
    }

}

// Epanechnikov kernel
__global__ void epane_kernel(double *pdfset, double *coordo,
                       double *coordt, double *bd,
                       int No, int Nt, int nvar)
{
    double pdf;                               // the pdf of the ith location
    double prod_kern;                              // the kernel information of the ith location
    double u, kernel;                              // the kernel information of the ith location in the kth variable
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the index of the ith location
    // int xi;                                        // the ith location

    /* if (i >= blockDim.x*gridDim.x) { */
    /*   return; */
    /* } */

    while (i < Nt) {
      pdf = 0.;
      for (int j = 0; j < No; j++)
      {
        prod_kern = 1.;
        for (int k = 0; k < nvar; k++)
        {
          u  = (coordt[i*nvar+k] - coordo[j*nvar+k]) / bd[k];

          if (u*u < 1)  // the Epanechnikov kernel
          {
            kernel = 0.75 * (1-u*u);
            prod_kern = prod_kern * kernel/bd[k];
          } else {
            prod_kern = 0.;
            break;
          }
        }
        // if (prod_kern > 0) printf("prod_kern %f\n", prod_kern);
        pdf = pdf + prod_kern;
      }

      pdfset[i] = pdf/(double)No;

      // Roll the block to see whether (for 1D)
      i += blockDim.x*gridDim.x;
    }

}


/**
 * [cuda_kde kernel density function]
 * @param  nvar   [number of variables]
 * @param  Nt     [number of pdf to be estimated]
 * @param  No     [number of the given samples]
 * @param  ktype  [the type of kernel]
 * @param  bd     [an array of bandwidths of the kernel]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
extern "C" {
double *cuda_kde(int nvar, int Nt, int No, int ktype, double *bd, double *coordo, double *coordt)
{
  /* int    blockWidth = 512;           // number of thread in each block */
  /* int    blocksX = Nt/blockWidth+1;  // number of block */
  double *pdfset, *d_pdfset;         // the pdf array to be estimated
  double *d_coordo, *d_coordt;       // the coordo and coordt in GPU memory
  double *d_bd;                      // the bd in GPU memory
  // double d_bd, d_No;
  /* double start = get_time(); */
  /* double stop; */

  // Allocate the host memory to pdfset
  pdfset = (double*)malloc(Nt*sizeof(double));
  //   datat = new double(2*N);

  // Allocate the device memory
  cudaMalloc(&d_pdfset, Nt*sizeof(double));
  cudaMalloc(&d_coordt, nvar*Nt*sizeof(double));
  cudaMalloc(&d_coordo, nvar*No*sizeof(double));
  cudaMalloc(&d_bd, nvar*sizeof(double));

  // Copy the host to the device
  cudaMemcpy(d_coordt, coordt, nvar*Nt*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coordo, coordo, nvar*No*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bd, bd, nvar*sizeof(double), cudaMemcpyHostToDevice);

  // Set d_pdfset to zero
  cudaMemset(d_pdfset, 0, Nt*sizeof(double));

  // Perform cuda KDE
  /* const dim3 block_size(blockWidth, 1, 1); */
  /* const dim3 grid_size(blocksX, 1, 1); */
  if(ktype == 1) {
    gaussian_kernel<<<512,512>>>(d_pdfset, d_coordo, d_coordt, d_bd, No, Nt, nvar);
  }
  else if(ktype == 2){
    epane_kernel<<<512,512>>>(d_pdfset, d_coordo, d_coordt, d_bd, No, Nt, nvar);
  }
  else {
    printf("Unknown kernel type index: %d", ktype);
  }
  /* kernel<<<100,100>>>(d_pdfset, d_coordo, d_coordt, d_bd, No, Nt, nvar); */

  // Copy the device back to the host
  cudaMemcpy(pdfset, d_pdfset, Nt*sizeof(double), cudaMemcpyDeviceToHost);

  // Free the memory
  cudaFree(d_coordo);
  cudaFree(d_pdfset);
  cudaFree(d_coordt);
  cudaFree(d_bd);

  /* stop = get_time(); */
  /* printf("CUDA time usage= %15.12f\n", stop - start); */

  // int c = 0;
  // int c2 = 0;
  // for (size_t i = 0; i < nvar*Nt; i++) {
  //   // printf("%f\n", coordo[i]);
  //   if (coordt[i] > 0) c2++;
  // }
  // for (size_t i = 0; i < nvar*No; i++) {
  //   // printf("%f\n", coordo[i]);
  //   if (coordo[i] > 0) c++;
  // }
  // printf("number of positive sampled: %d\n", c);
  // printf("number of positive estimated: %d\n", c2);

  return pdfset;
}
}
