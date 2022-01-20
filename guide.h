#pragma once

#include "defines.h"
#include <memory>
#include <vector>

struct matrixIndexPair
{
	int lowIndex;
	int highIndex;
};

struct reductionGuide
{
	int increment;
	bool handleOdd;
};


class guide
{
	private:
		void setupReductionHelper(std::vector<reductionGuide> &guide, int size, int reductions, int& threads);

	public:
		guide(int permutation_size, int matrix_size, int rows);

		std::vector<matrixIndexPair> constructionHelper;
		std::vector<reductionGuide> summationHelper;
		std::vector<reductionGuide> maximaHelper;

		int summation_threads;
		int summation_reductions;

		int maxima_threads;
		int maxima_reductions;

		int matrix_size;

};

