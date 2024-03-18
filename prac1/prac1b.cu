//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}


__global__ void vecAdd(float *x, float *y, float *z)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  x[tid] = (float) threadIdx.x;
  y[tid] = (float) 2*threadIdx.x;
  z[tid] = x[tid] + y[tid]; 
	
}
//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x, *h_x2, *d_x2, *h_x3, *d_x3 ;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 1;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  h_x2 = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x2, nsize*sizeof(float)));
  h_x3 = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x3, nsize*sizeof(float)));
  // execute kernel
  
  // my_first_kernel<<<nblocks,nthreads>>>(d_x);
  // my_first_kernel<<<nblocks,nthreads>>>(d_x2);
  vecAdd<<<nblocks,nthreads>>>(d_x, d_x2, d_x3);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  checkCudaErrors( cudaMemcpy(h_x2,d_x2,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(h_x3,d_x3,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );
  
  for (n=0; n<nsize; n++) 
  {
	printf(" n,  x  =  %d  %f \n",n,h_x[n]);
	printf(" n,  y  =  %d  %f \n",n,h_x2[n]);
	printf(" n,  z  =  %d  %f \n",n,h_x3[n]);
  }
  // free memory 

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_x2));
  checkCudaErrors(cudaFree(d_x3));
  free(h_x);
  free(h_x2);
  free(h_x3);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
