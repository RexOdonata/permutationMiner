#include "guide.h"

#include <math.h>

guide::guide(int permutation_size, int matrix_size, int rows)
{
	// the amount of reduction iterations to be performed to sum the differences in each row
	summation_reductions = ceil(log2(matrix_size));

	// the total number of items to reduced ( 0 padded to a power of 2)
	summation_size = pow(2, summation_reductions);

	// the amount of reduction iterations to be performed to find the maximum difference from all rows
	maxima_reductions = ceil(log2((double)rows));

	// the amount of reduction iterations to be performed to find the maximum difference from all rows
	maxima_size = pow(2, maxima_reductions);

	for (int i = 0; i < permutation_size; i++)
	{
		for (int j = i + 1; j < permutation_size; j++)
		{
			matrixIndexPair newPair{i,j};
			constructionHelper.push_back(newPair);
		}
	}


	// setting up the summation guide
	setupReductionHelper(summationHelper, summation_size, summation_threads);
	

	// setting up the maxima finder guide	
	setupReductionHelper(maximaHelper, maxima_size, maxima_threads);
}


void guide::setupReductionHelper(std::vector<int>& guide, int size, int &threads)
{
	//fill the array with guides
	//guides show each element how far to look ahead and whehter or not the current iteration needs to handle odd
	for (int i = size; 1 < i; i = i / 2)
	{
		guide.push_back(i/2);
	}
	threads = guide.front();
}