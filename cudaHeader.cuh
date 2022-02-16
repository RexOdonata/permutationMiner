#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "guide.h"
#include "defines.h"


void construct(keyEntry * permutation, keyEntry * matrix, matrixIndexPair * ConstructionGuide, const int permutation_size, const int  matrix_size, const int rows);

__global__ void device_Construct(keyEntry* permutation, keyEntry* matrix, matrixIndexPair* ConstructionGuide, const int permutation_size);

void difference(keyEntry * matrix, keyEntry * baseMatrix, const int matrix_size, const int rows);

__global__ void device_difference(keyEntry* matrix, keyEntry* baseMatrix);

void summation(keyEntry* matrix, keyEntry * rowSums, int * incGuide, const int reduction_size, const int matrix_size, const int reductions, const int rows, const int threads);

__global__ void device_summation(keyEntry * matrix, keyEntry *  rowSums, int * incGuide, const int reduction_size, const int matrix_size, const int reductions);

void maxima(keyEntry* rowSums, keyEntry* gpu_max, int* incGuide, const int matrix_size, const int reduction_size,  const int reductions, int threads);

__global__ void device_maxima(keyEntry* rowSums, keyEntry* gpu_max, int * incGuide, const int reduction_size, const int reductions, const int matrix_size);