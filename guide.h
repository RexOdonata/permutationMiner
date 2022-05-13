#pragma once

#include "defines.h"
#include <memory>
#include <vector>

struct matrixIndexPair
{
	int lowIndex;
	int highIndex;
};


class guide
{
	private:
		void setupReductionHelper(std::vector<int> &guide, int size, int& threads);

	public:
		guide(int permutation_size, int matrix_size, int rows);

		std::vector<matrixIndexPair> constructionHelper;
		std::vector<int> summationHelper;
		std::vector<int> maximaHelper;

		// how many threads to use in a summation reduction
		int summation_threads;
		// how many reduction steps to perform in a summation reduction
		int summation_reductions;
		//the total number of items to reduced (always a power of 2)
		int summation_size;

		int maxima_threads;
		int maxima_reductions;
		int maxima_size;

		int matrix_size;

};

