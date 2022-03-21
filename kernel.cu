
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaHeader.cuh"



void construct(keyEntry* permutation, keyEntry* matrix, keyEntry* baseMatrix, matrixIndexPair * constructionGuide, const int permutation_size, const int matrix_size, const int rows)
{
	device_Construct << < rows, matrix_size >> > (permutation, matrix, baseMatrix, constructionGuide, permutation_size);
}

__global__ void device_Construct(keyEntry* permutation, keyEntry* matrix, keyEntry* baseMatrix, matrixIndexPair* ConstructionGuide, const int permutation_size)
{
	keyEntry fIndex = blockIdx.x * (permutation_size) + ConstructionGuide[threadIdx.x].lowIndex;
	keyEntry sIndex = blockIdx.x * (permutation_size) + ConstructionGuide[threadIdx.x].highIndex;

	keyEntry fVal = permutation[fIndex];
	keyEntry sVal = permutation[sIndex];


	unsigned short row, col;

	// ensure that column always gets the higher number and row gets the lower number
	// the ternary operators are slight performance improvement over an if-else
	// I think this is because they are able to compile to max-load instructions which don't result in divergence
	row = min(fVal, sVal);
	col = max(fVal, sVal);

	row--;
	col--;

	int	outputIndex = col - 1 + (row * (permutation_size - 2) - ((row - 1) * row) / 2);
	keyEntry valM = ConstructionGuide[threadIdx.x].highIndex - ConstructionGuide[threadIdx.x].lowIndex;
	keyEntry valB = baseMatrix[outputIndex];
	matrix[blockIdx.x * blockDim.x + outputIndex] = abs(valM-valB);
}

void summation(keyEntry* matrix, keyEntry* rowSums, int* incGuide, const int reduction_size, const int matrix_size, const int reductions, const int rows, const int threads)
{
	device_summation << < rows, threads, reduction_size * sizeof(keyEntry) >>> (matrix, rowSums, incGuide, reduction_size, matrix_size, reductions);
}

__global__ void device_summation(keyEntry* matrix, keyEntry* rowSums, int* incGuide, const int reduction_size, const int matrix_size, const int reductions)
{
	extern __shared__ keyEntry reductionData[];

	//transfer from global to shared memory in two halves
	//for the second half, pad with zeroes where neccessary.
	reductionData[threadIdx.x] = matrix[blockIdx.x * (matrix_size)+threadIdx.x];
	reductionData[threadIdx.x + reduction_size / 2] = (matrix_size-1 < threadIdx.x + reduction_size / 2) ? 0 : matrix[blockIdx.x * (matrix_size)+threadIdx.x + reduction_size / 2];

	__syncthreads();

	for (int i = 0; i < reductions; i++)
	{
		if (threadIdx.x < incGuide[i]) reductionData[threadIdx.x] += reductionData[threadIdx.x + incGuide[i]];
		__syncthreads();
	}

	if (threadIdx.x == 0) rowSums[blockIdx.x] = reductionData[0];
	
}

void maxima(keyEntry* rowSums, keyEntry* gpu_max, int* guide, const int rows, const int reduction_size, const int reductions, int threads)
{
	device_maxima << < 1, threads, reduction_size * sizeof(keyEntry) >> > (rowSums, gpu_max, guide, reduction_size, reductions, rows);
}

__global__ void device_maxima( keyEntry* rowSums, keyEntry* gpu_max, int* incGuide, const int reduction_size, const int reductions, const int rows)
{
	extern __shared__ keyEntry reductionData[];

	reductionData[threadIdx.x] = rowSums[threadIdx.x];
	reductionData[threadIdx.x + reduction_size / 2] = (rows - 1 < threadIdx.x + reduction_size / 2) ? 0 : rowSums[threadIdx.x + reduction_size / 2];

	__syncthreads();

	for (int i = 0; i < reductions; i++)
	{
		if (threadIdx.x < incGuide[i]) reductionData[threadIdx.x] = max(reductionData[threadIdx.x],reductionData[threadIdx.x + incGuide[i]]);
		__syncthreads();
	}

	if (threadIdx.x == 0) *gpu_max = max(*gpu_max,reductionData[0]);
}