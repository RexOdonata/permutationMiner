#pragma once
#include "defines.h"
#include "feeder.h"
#include "generator.h"
#include "guide.h"
#include "gpuMem.h"

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
#include <mutex>

class feeder;

class processor
{
	public:
		processor(int set_permutation_Size, std::vector<keyEntry>& set_preset);

		void run();

		void processFrame(std::vector<keyEntry> &input, const char feedID, cudaStream_t stream, gpuFeedMem feedMem);

		const keyEntry getMax();

		void printCompletion(const std::string msg);

		static void cudaErrorCheck(const std::string msg);

	private:

		//functions
		void createFeeds();

		void runLoop();

		const void printData(const int frameNum, keyEntry * data);

		void initGPUMemory();

		//basic members
		const int permutation_size;
		const int matrix_size;
		const int rows = 1024;

		const int feedThreads = FEEDS;

		int data_size;

		keyEntry maxDifference = 0;

		//object members
		std::vector<keyEntry> preset_permutation;

		std::array<std::unique_ptr<feeder>,FEEDS> factory;


		std::unique_ptr<guide> helper;

		std::mutex printMtx;

		gpuCommonMem gpu_commons;


		

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
		const void display_TCT();
#endif
};



