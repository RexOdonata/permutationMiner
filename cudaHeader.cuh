#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "guide.h"
#include "defines.h"


void construct(keyEntry * permutation, keyEntry * matrix, matrixIndexPair * ConstructionGuide, int * permutation_size, int  matrix_size, int rows);

__global__ void device_Construct(keyEntry* permutation, keyEntry* matrix, matrixIndexPair* ConstructionGuide, int * permutationSize);

void difference(keyEntry * matrix, keyEntry * baseMatrix, int matrix_size, int rows);

__global__ void device_difference(keyEntry* matrix, keyEntry* baseMatrix);

void summation(keyEntry* matrix, keyEntry * rowSums, reductionGuide * guide, int * matrix_size, int * reductions, int rows, int threads);

__global__ void device_summation(keyEntry * matrix, keyEntry *  rowSums, reductionGuide * guide, int * matrix_size, int * reductions);

void maxima(keyEntry* rowSums, reductionGuide* guide, int * gpu_matrix_size, keyEntry * gpu_max, int * reductions, int threads);

__global__ void device_maxima(keyEntry* rowSums, reductionGuide* guide, int* reductions, keyEntry * gpu_max, int * gpu_matrix_size);