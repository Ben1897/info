/**
* @Author: Peishi Jiang <Ben1897>
* @Date:   2017-02-27T10:13:51-06:00
* @Email:  shixijps@gmail.com
* @Last modified by:   Ben1897
* @Last modified time: 2017-02-28T13:44:05-06:00
*/

/**
 * Include all the packages
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/**
 * [kde kernel density function
 *  Note that: to calculate multivariate kernel, the product kernel is employed.
 *  For details, please refer to:
 *  http://www.buch-kromann.dk/tine/nonpar/Nonparametric_Density_Estimation_multidim.pdf.]
 *
 * @param  nvar   [number of variables]
 * @param  Nt     [number of pdf to be estimated]
 * @param  No     [number of the given samples]
 * @param  bd     [an array of bandwidths of the kernel]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
double *kde(int nvar, int Nt, int No, double *bd, double *coordo, double *coordt)
{
  // double xi;         // The ith location of the estimated point in the kth variable
  // double yj;         // The jth location of the sampled point in the kth variable
  double u, kernel;  // The kernel information of the ith location in the kth variable
  double prod_kern;  // The kernel information of the ith location
  double pdf;        // The pdf of the ith location
  double *pdfset;    // The pdf array to be estimated
  // time_t start,end;  // time

  // Allocate the host memory to pdfset
  pdfset = (double*)malloc(Nt*nvar*sizeof(double));

  ///////////////////////////
  // Estimate the pdf array
  //////////////////////////
  // start = clock();
  for (int i = 0; i < Nt; i++)
  {
    pdf = 0.;
    for (int j = 0; j < No; j++)
    {
      prod_kern = 1.;
      for (int k = 0; k < nvar; k++)
      {
        u  = (coordt[i*nvar+k] - coordo[j*nvar+k]) / bd[k];
        // Epanechnikov kernel
        /* if (u*u < 1) */
        /* { */
        /*   kernel = 0.75 * (1-u*u);  // the Epanechnikov kernel */
        /*   prod_kern = prod_kern * kernel/bd[k]; */
        /* } else { */
        /*   prod_kern = 0.; */
        /*   break; */
        /* } */

        // Gaussian kernel
        kernel = 1./sqrt(2*M_PI) * exp(-1./2.*pow(u,2));
        prod_kern = prod_kern * kernel/bd[k];
     }

      pdf = pdf + prod_kern;
    }

    pdfset[i] = pdf/(double)No;
  }
  // end = clock();
  // printf("the total number time executed: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

  return pdfset;
}
