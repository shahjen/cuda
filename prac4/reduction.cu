////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float *idata, int len)
{
	float sum = 0.0f;
	for (int i = 0; i < len; i++)
		sum += idata[i];

	return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata, const int nice_thread_size)
{
	// dynamically allocated shared memory -- shared by block

	extern __shared__ float temp[];

	int tid = threadIdx.x;// + blockDim.x*blockIdx.x;
	int global_id =  threadIdx.x + blockDim.x*blockIdx.x;
	// first, each thread loads data into shared memory

	temp[tid] = g_idata[global_id];
	__syncthreads();

	// handle non elegant thread size
	if(tid>=nice_thread_size)
		temp[tid-nice_thread_size]+=temp[tid];

	// next, we perform binary tree reduction
	for (int d = nice_thread_size/ 2; d > 0; d = d / 2)
	{
		__syncthreads(); // ensure previous step completed
		if (tid < d)
			temp[tid] += temp[tid + d];
	}

	// finally, first thread puts result into global memory
	// printf("TID : %d\n", tid);
	__syncthreads();
	if (tid == 0)
		g_odata[blockIdx.x] = temp[0];
}

////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
	int num_blocks, num_threads, num_elements, mem_size, shared_mem_size, num_threads_nice;

	float *h_data, *d_idata, *d_odata;

	// initialise card

	findCudaDevice(argc, argv);

	num_blocks = 2; // start with only 1 thread block
	num_threads = 514;
	num_elements = num_blocks * num_threads;
	mem_size = sizeof(float) * num_elements;

	num_threads_nice = 1<<((int)log2(num_threads));
	printf("Num threads nice = %d\n", num_threads_nice);
	printf("Mem size = %d\n", mem_size);
	// allocate host memory to store the input data
	// and initialize to integer values between 0 and 10

	h_data = (float *)malloc(mem_size);

	for (int i = 0; i < num_elements; i++)
		h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));

	// compute reference solution

	float sum = reduction_gold(h_data, num_elements);

	// allocate device memory input and output arrays

	checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
	checkCudaErrors(cudaMalloc((void **)&d_odata, num_blocks*sizeof(float)));

	// copy host memory to device input array

	checkCudaErrors(cudaMemcpy(d_idata, h_data, mem_size,
							   cudaMemcpyHostToDevice));

	// execute the kernel

	shared_mem_size = sizeof(float) * num_threads;
	reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata, num_threads_nice);
	getLastCudaError("reduction kernel execution failed");

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_data, d_odata, num_blocks*sizeof(float),
							   cudaMemcpyDeviceToHost));

	// check results
	// method - 1 global reduce
	float sum_parallel{0.0f};
	for(int i=0;i<num_blocks;i++)
		sum_parallel+=h_data[i];
	printf("reduction error = %f -%f = %f\n", sum_parallel, sum, sum_parallel - sum);

	// cleanup memory

	free(h_data);
	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));

	// CUDA exit -- needed to flush printf write buffer

	cudaDeviceReset();
}
