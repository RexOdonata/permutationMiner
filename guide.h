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

		int summation_threads;
		int summation_reductions;
		int summation_size;

		int maxima_threads;
		int maxima_reductions;
		int maxima_size;

		int matrix_size;

};

