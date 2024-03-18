////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

__constant__ int N;
__constant__ float a, b, c;

__global__ void eval_quad_mean(float *d_z, float *d_v)
{
    int idx;
    float val{0.0f}, eval{0.0f};

    idx = threadIdx.x + N*blockIdx.x*blockDim.x;
    for (int n=0; n<N; n++)
    {
        val = d_z[idx];
        eval += a*val*val + b*val + c;
        idx += blockDim.x;      // shift pointer to next element
    }
    d_v[threadIdx.x + blockIdx.x*blockDim.x] = eval/N;
}

int main(int argc, const char **argv)
{    
    int     NPATH=9600000, h_N=200; // why not #define here?
    float   h_a{1.2f}, h_b{0.7f}, h_c(1.98f);
    float  *h_v, *d_v, *d_z;
    double  sum1{0.0};

    // initialise card

    findCudaDevice(argc, argv);

    // initialise CUDA timing

    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on host and device

    h_v = (float *)malloc(sizeof(float)*NPATH);

    checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
    checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );

    // define constants and transfer to GPU
    checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
    checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
    checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
    checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );
    // random number generation

    curandGenerator_t gen;
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

    cudaEventRecord(start);
    checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
            milli, h_N*NPATH/(0.001*milli));

    // execute kernel and time it

    cudaEventRecord(start);
    eval_quad_mean<<<NPATH/128, 128>>>(d_z, d_v);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    getLastCudaError("eval_quad execution failed\n");
    printf("Monte Carlo kernel execution time (ms): %f \n",milli);

    // copy back results

    checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                    cudaMemcpyDeviceToHost) );

    // compute average

    for (int i=0; i<NPATH; i++)
        sum1 += h_v[i];

    printf("\nAverage value and standard deviation of error  = %13.8f\n\n",
        sum1/NPATH );

    // Tidy up library

    checkCudaErrors( curandDestroyGenerator(gen) );

    // Release memory and exit cleanly

    free(h_v);
    checkCudaErrors( cudaFree(d_v) );
    checkCudaErrors( cudaFree(d_z) );

    // CUDA exit -- needed to flush printf write buffer

    cudaDeviceReset();

}