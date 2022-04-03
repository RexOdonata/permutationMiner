#pragma once

#include "processor.h"
#include "generator.h"
#include "gpuMem.h"

#include <memory>
#include <chrono>

#include "cuda_runtime.h"


class processor;

class feeder
{
	public:
		feeder(int set_rowsPerThread, int set_permutation_size, processor * set_parent, char id);

		void storeGenerator(std::unique_ptr<generator> in);

		void loadGenerator();

		void run();

		const keyEntry getMax();

#if frameTiming
		std::vector<double> frameFillTime;
		std::chrono::time_point<std::chrono::high_resolution_clock> fTic;
		void clock_FPT();
		void record_FPT();
		void display_FPT();

#endif // 

	private:
		
		const int rows;
		const int permutation_size;
		const int matrix_size;
		const char id;

		keyEntry max;

		gpuFeedMem gpu_feed_private;

		cudaStream_t stream;

		std::vector<keyEntry> data;

		bool done = false;

		processor * proc;

		std::unique_ptr<generator> source;

		std::vector<std::unique_ptr<generator>> seeds;

		bool fillFrame();

		void padFrame(int startRow);





};

