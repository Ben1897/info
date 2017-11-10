/**
 * Kernel Density Estimation Implementation, including Gaussian kernels
 * with general kernels for multivariate density estimation
 *
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
 * @param bdinv  [an array representing the inverse bandwidth matrix]
 * @param bddet  [the determinant of the bandwidth matrix]
 * @param No     [number of the given samples]
 * @param Nt     [number of pdf to be estimated]
 * @param nvar   [number of variables]
 */

// Gaussian kernel
__global__ void gaussian_kernel(double *pdfset, double *coordo,
                                double *coordt, double *bdinv, double bddet,
                                int No, int Nt, int nvar)
{
    double pdf;                               // the pdf of the ith location
    double kernel;                                 // the kernel information of the ith location in the kth variable
    double xbdinv, xbdinvxT;                       // the mutplication of the dist, bdinv and the transpose of dist
    double distk;                                  // the distance between two vectors in the kth variable
    int i = blockIdx.x * blockDim.x + threadIdx.x; // the index of the ith location

    while (i < Nt) {
      pdf = 0.;
      for (int j = 0; j < No; j++)
      {
        /* Compute the mutplication of the dist, coninv and the transpose of dist */
        xbdinvxT = 0.;
        for (int ki = 0; ki < nvar; ki++)
        {
          xbdinv = 0.;
          for (int kj = 0; kj < nvar; kj++)
          {
            distk  = coordt[i*nvar+kj] - coordo[j*nvar+kj];
            xbdinv = xbdinv + distk*bdinv[kj*nvar+ki];
          }

          distk  = coordt[i*nvar+ki] - coordo[j*nvar+ki];
          xbdinvxT = xbdinvxT + distk*xbdinv;
        }

        /* Compute the kernel */
        kernel = 1./sqrt(pow(2*M_PI,nvar)*bddet) * exp(-1./2.*xbdinvxT);

        pdf = pdf + kernel;
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
 * @param  bdinv  [an array representing the inverse bandwidth matrix]
 * @param  bddet  [the determinant of the bandwidth matrix]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
extern "C" {
  double *cuda_kde_general(int nvar, int Nt, int No, double bddet, double *bdinv, double *coordo, double *coordt)
  {
    double *pdfset, *d_pdfset;         // the pdf array to be estimated
    double *d_coordo, *d_coordt;       // the coordo and coordt in GPU memory
    double *d_bdinv;                      // the bd in GPU memory

    // Allocate the host memory to pdfset
    pdfset = (double*)malloc(Nt*sizeof(double));

    // Allocate the device memory
    cudaMalloc(&d_pdfset, Nt*sizeof(double));
    cudaMalloc(&d_coordt, nvar*Nt*sizeof(double));
    cudaMalloc(&d_coordo, nvar*No*sizeof(double));
    cudaMalloc(&d_bdinv, nvar*nvar*sizeof(double));

    // Copy the host to the device
    cudaMemcpy(d_coordt, coordt, nvar*Nt*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coordo, coordo, nvar*No*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bdinv, bdinv, nvar*nvar*sizeof(double), cudaMemcpyHostToDevice);

    // Set d_pdfset to zero
    cudaMemset(d_pdfset, 0, Nt*sizeof(double));

    // Perform cuda KDE
    gaussian_kernel<<<512,512>>>(d_pdfset, d_coordo, d_coordt, d_bdinv, bddet, No, Nt, nvar);

    // Copy the device back to the host
    cudaMemcpy(pdfset, d_pdfset, Nt*sizeof(double), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_coordo);
    cudaFree(d_pdfset);
    cudaFree(d_coordt);
    cudaFree(d_bdinv);

    /* for (int k = 0; k < nvar; k++) */
    /* { */
    /*   printf("%.4f \n", bdinv[k*nvar+k]); */
    /*   printf("%.4f \n", 1./sqrt(bdinv[k*nvar+k])); */
    /* } */
    /* printf("%d  %d", No, Nt); */
    /* printf("%.6f", pdfset[0]); */

    return pdfset;
  }
}
