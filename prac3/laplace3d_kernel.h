//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//

// device code

// __constant__ float sixth=1.0f/6.0f;
__global__ void GPU_laplace3d(long long NX, long long NY, long long NZ,
							const float *__restrict__ d_u1,
							  float *__restrict__ d_u2,
							  float *d_u3)
{
	long long i, j, k, indg, IOFF, JOFF, KOFF;
	int tid{threadIdx.x+blockDim.x*threadIdx.y};
	float u2, sixth = 1.0f / 6.0f, squared_diff{0.0f};

	//
	// define global indices and array offsets
	//

	i = threadIdx.x + blockIdx.x * BLOCK_X;
	j = threadIdx.y + blockIdx.y * BLOCK_Y;
	indg = i + j * NX;

	IOFF = 1;
	JOFF = NX;
	KOFF = NX * NY;

	if (i >= 0 && i <= NX - 1 && j >= 0 && j <= NY - 1)
	{

		for (k = 0; k < NZ; k++)
		{

			if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
			{
				u2 = d_u1[indg]; // Dirichlet b.c.'s
			}
			else
			{
				u2 = (d_u1[indg - IOFF] + d_u1[indg + IOFF] + d_u1[indg - JOFF] + d_u1[indg + JOFF] + d_u1[indg - KOFF] + d_u1[indg + KOFF]) * sixth;
			}
			squared_diff+=(u2-d_u2[indg])*(u2-d_u2[indg]);
			d_u2[indg] = u2;
			indg += KOFF;
		}
	}
	__syncthreads();
	for(int d=warpSize/2; d>0; d=d/2)
		squared_diff += __shfl_down_sync(-1, squared_diff, d);
	__syncthreads();
	if (0==tid%warpSize)
		atomicAdd(&d_u3[0], squared_diff);
}
