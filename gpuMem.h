#pragma once

#include "defines.h"
#include "guide.h"

struct gpuFeedMem
{
	//gpu permutation memory
	keyEntry* gpu_permutation_data;
	//gpu UTM memory
	keyEntry* gpu_matrix_UTM;
	//gpu row summary array
	keyEntry* gpu_row_sums;

	//gpu  maxima result
	keyEntry* gpu_constant_maxima;
};

struct gpuCommonMem
{
	//gpu baseMatrix
	keyEntry* gpu_matrix_base;

	//gpu matrix construction guide
	matrixIndexPair* gpu_guide_construction;
	//gpu summation reduction guide
	int* gpu_guide_summation;
	//gpu maximization reduction guide
	int* gpu_guide_maxima;
};
