#include "guide.h"

#include <math.h>

guide::guide(int permutation_size, int matrix_size, int rows)
{
	// the amount of reduction iterations to be performed to sum the differences in each row
	summation_reductions = (int)ceil(log2((double)matrix_size));

	// the amount of reduction iterations to be performed to find the maximum difference from all rows
	maxima_reductions = (int)ceil(log2((double)rows));

	for (int i = 0; i < permutation_size; i++)
	{
		for (int j = i + 1; j < permutation_size; j++)
		{
			matrixIndexPair newPair{i,j};
			constructionHelper.push_back(newPair);

		}
	}


	// setting up the summation guide
	setupReductionHelper(summationHelper, matrix_size, summation_reductions, summation_threads);
	

	// setting up the maxima finder guide	
	setupReductionHelper(maximaHelper, rows, maxima_reductions, maxima_threads);
}


void guide::setupReductionHelper(std::vector<reductionGuide>& guide, int size, int reductions, int &threads)
{
	//fill the array with guides
	//guides show each element how far to look ahead and whehter or not the current iteration needs to handle odd
	auto counter = size;
	for (int i = 0; i < reductions; i++)
	{
		reductionGuide newGuide;
		newGuide.increment = counter / 2;
		// if the increment is odd
		if (counter % 2 > 0)
		{
			newGuide.handleOdd = true;
		}
		else
			newGuide.handleOdd = false;

		guide.push_back(newGuide);
		counter -= newGuide.increment;
	}

	if (size % 2 > 0) threads = guide[0].increment + 1;
	else threads = guide[0].increment;
}