#pragma once
#include "defines.h"
#include "feeder.h"
#include "generator.h"
#include "guide.h"

#include "cudaHeader.cuh"

#include <vector>
#include <thread>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>
#include <chrono>
#include <array>
#include <atomic>

class feeder;

class processor
{
	public:
		processor(int set_permutation_Size, std::vector<keyEntry>& set_preset);

		void run();

		void contactProcessor(std::vector<keyEntry> &input, char);

		void relaxLock();

		keyEntry getMax();

	private:

		//functions
		void createFeeds();

		keyEntry processDataFrame(std::vector<keyEntry>& input);

		void printData(int frameNum, keyEntry * data);

		void initGPUMemory();

		

		//basic members
		const int permutation_size;
		const int matrix_size;
		int rowsPerThread;

		int data_size;

		keyEntry maxDifference = 0;

		//object members
		std::vector<keyEntry> preset_permutation;

		std::array<std::unique_ptr<feeder>,2> factory;

		std::atomic_char gpuLock;
		char lockBreaker = 0;

		std::unique_ptr<guide> helper;

		//gpu permutation memory
		keyEntry * gpu_permutation_data;
		//gpu UTM memory
		keyEntry * gpu_matrix_UTM;
		//gpu row summary array
		keyEntry* gpu_row_sums;
		//gpu permutation size
		int * gpu_constant_permutationSize;
		//gpu permutation size
		int* gpu_constant_matrixSize;
		//gpu summation reductions
		int * gpu_constant_sumReductions;
		//gpu maxima reductions
		int * gpu_constant_maxReductions;
		//gpu  maxima result
		keyEntry * gpu_constant_maxima;
		//gpu baseMatrix
		keyEntry * gpu_matrix_base;

		//gpu matrix construction guide
		matrixIndexPair * gpu_guide_construction;
		//gpu summation reduction guide
		reductionGuide * gpu_guide_summation;
		//gpu maximization reduction guide
		reductionGuide * gpu_guide_maxima;

#if frameTiming
		std::vector<double> frameCompTime;
		std::chrono::time_point<std::chrono::high_resolution_clock> fTic;
		void clock_FCT();
		void record_FCT();
		void display_FCT();
#endif

#if runTiming
		std::chrono::time_point<std::chrono::high_resolution_clock> tTic;
		std::chrono::duration<double> timeElapsed;
		void clock_TCT();
		void record_TCT();
		void display_TCT();
#endif
};



