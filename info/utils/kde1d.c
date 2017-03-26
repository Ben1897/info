/**
* @Author: Peishi Jiang <Ben1897>
* @Date:   2017-02-27T10:13:51-06:00
* @Email:  shixijps@gmail.com
* @Last modified by:   Ben1897
* @Last modified time: 2017-02-28T13:44:14-06:00
*/

/**
 * Include all the packages
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * [kde 1D kernel density function]
 * @param  bd     [bandwidth of the kernel]
 * @param  Nt     [number of pdf to be estimated]
 * @param  No     [number of the given samples]
 * @param  coordo [an array of the sample locations]
 * @param  coordt [an array of the location of pdf to be estimated]
 * @return        [an array of pdf to be estimated]
 */
double *kde(double bd, int Nt, int No, double *coordo, double *coordt)
{
  double pdf;
  double u, kernel;
  double *pdfset;    // The pdf array to be estimated
  // time_t start,end;

  // allocate the host memory to pdfset
  pdfset = (double*)malloc(Nt*sizeof(double));
  //   datat = new double(2*N);

  // Estimate the pdf array
  // start = clock();
  for (int i = 0; i < Nt; i++) {
    pdf = 0.;

    for (int j = 0; j < No; j++) {
      u = (coordt[i] - coordo[j]) / bd;
      if (u*u < 1) {
        kernel = 0.75 * (1-u*u);  // the Epanechnikov kernel
        pdf = pdf + kernel/bd;
      }
    }

    pdfset[i] = pdf/(double)No;
  }
  // end = clock();
  // printf("the total number time executed: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

  return pdfset;
}
