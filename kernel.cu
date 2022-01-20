
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaHeader.cuh"

#include <cooperative_groups.h>

using namespace cooperative_groups;


void construct(keyEntry* permutation, keyEntry* matrix, matrixIndexPair * ConstructionGuide, int * permutation_size, int matrix_size, int rows)
{
	device_Construct << < rows, matrix_size >> > (permutation, matrix, ConstructionGuide, permutation_size);
}

__global__ void device_Construct(keyEntry* permutation, keyEntry* matrix, matrixIndexPair* ConstructionGuide, int * permutation_size)
{
	auto fIndex = blockIdx.x * (*permutation_size) + ConstructionGuide[threadIdx.x].lowIndex;
	auto sIndex = blockIdx.x * (*permutation_size) + ConstructionGuide[threadIdx.x].highIndex;
	keyEntry fVal = permutation[fIndex];
	keyEntry sVal = permutation[sIndex];

	unsigned short row, col;

	if (fVal > sVal)
	{
		row = sVal - 1;
		col = fVal - 1;
	}
	else
	{
		col = sVal - 1;
		row = fVal - 1;
	}

	unsigned short	outputIndex = col - 1 + (row * (*permutation_size - 2) - ((row - 1) * row) / 2);

	matrix[blockIdx.x * blockDim.x + outputIndex] = ConstructionGuide[threadIdx.x].highIndex - ConstructionGuide[threadIdx.x].lowIndex;
}

void difference(keyEntry* matrix, keyEntry* baseMatrix, int matrix_size, int rows)
{
	device_difference << < rows, matrix_size >> > (matrix, baseMatrix);
}

__global__ void device_difference(keyEntry* matrix, keyEntry* baseMatrix)
{
	keyEntry valM = matrix[blockIdx.x * blockDim.x + threadIdx.x];
	keyEntry valB = baseMatrix[threadIdx.x];
	if (valB < valM)	matrix[blockIdx.x * blockDim.x + threadIdx.x] = valM - valB;
	else				matrix[blockIdx.x * blockDim.x + threadIdx.x] = valB - valM;
}

void summation(keyEntry* matrix, keyEntry * rowSums, reductionGuide* guide, int * matrix_size,  int * reductions, int rows, int threads)
{
	device_summation << < rows, threads >>> (matrix, rowSums, guide, matrix_size, reductions);
}

__global__ void device_summation(keyEntry* matrix, keyEntry * rowSums, reductionGuide* guide, int * matrix_size, int* reductions)
{
	for (int i = 0; i < *reductions; i++)
	{
		if (guide[i].handleOdd == false && threadIdx.x < guide[i].increment)
		{
			matrix[blockIdx.x * (*matrix_size) + threadIdx.x] += matrix[blockIdx.x * (*matrix_size) + threadIdx.x + guide[i].increment];
		}			
		else if (guide[i].handleOdd == true && threadIdx.x <= guide[i].increment && !(threadIdx.x == 0))
		{
			matrix[blockIdx.x * (*matrix_size) + threadIdx.x] += matrix[blockIdx.x * (*matrix_size) + threadIdx.x + guide[i].increment];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) rowSums[blockIdx.x] = matrix[blockIdx.x * (*matrix_size)];
}

void maxima(keyEntry* rowSums, reductionGuide* guide, int * gpu_matrix_size, keyEntry* gpu_max, int * reductions, int threads)
{
	device_maxima << < 1, threads >> > (rowSums, guide, reductions, gpu_max, gpu_matrix_size);
}

__global__ void device_maxima(keyEntry* rowSums, reductionGuide* guide, int* reductions, keyEntry * gpu_max, int * gpu_matrix_Size)
{
	for (int i = 0; i < *reductions; i++)
	{
		if (guide[i].handleOdd == false && threadIdx.x < guide[i].increment)
		{
			rowSums[threadIdx.x] = max(rowSums[threadIdx.x], rowSums[threadIdx.x + guide[i].increment]);
		}
		else if (guide[i].handleOdd == true && threadIdx.x <= guide[i].increment && !(threadIdx.x == 0))
		{
			rowSums[threadIdx.x] = max(rowSums[threadIdx.x], rowSums[threadIdx.x + guide[i].increment]);
		}
		__syncthreads();
	}
	if (threadIdx.x==0) gpu_max[0] = rowSums[0];
}