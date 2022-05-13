#include "processor.h"

processor::processor(int set_permutation_Size, std::vector<keyEntry>& set_preset) :
	permutation_size(set_permutation_Size), matrix_size(set_permutation_Size* (set_permutation_Size - 1) / 2), preset_permutation(set_preset)
{
	createFeeds();

	helper = std::make_unique<guide>(permutation_size, matrix_size, rows);

	initGPUMemory();

}

void processor::initGPUMemory()
{

	// create the base matrix and load it
	{
		std::vector<keyEntry> base(matrix_size);
		auto start = 0;
		for (int i = permutation_size - 1; i > 0; i--)
		{
			std::iota(base.begin() + start, base.begin() + start + i, 1);
			start += i;
		}

		cudaMalloc((void**)&(gpu_commons.gpu_matrix_base), matrix_size * sizeof(keyEntry));
		processor::cudaErrorCheck("base matrix allocate");
		cudaMemcpy(gpu_commons.gpu_matrix_base, base.data(), matrix_size * sizeof(keyEntry), cudaMemcpyHostToDevice);
		processor::cudaErrorCheck("base matrix copy");
	}

	// allocate and load helpers
	{
		cudaMalloc((void**)&(gpu_commons.gpu_guide_construction), helper->constructionHelper.size() * sizeof(matrixIndexPair));
		processor::cudaErrorCheck("construction guide allocate");
		cudaMemcpy(gpu_commons.gpu_guide_construction, helper->constructionHelper.data(), helper->constructionHelper.size() * sizeof(matrixIndexPair), cudaMemcpyHostToDevice);
		processor::cudaErrorCheck("construction guide copy");

		cudaMalloc((void**)&(gpu_commons.gpu_guide_summation), helper->summationHelper.size() * sizeof(int));
		processor::cudaErrorCheck("summation guide allocate");
		cudaMemcpy(gpu_commons.gpu_guide_summation, helper->summationHelper.data(), helper->summationHelper.size() * sizeof(int), cudaMemcpyHostToDevice);
		processor::cudaErrorCheck("summation guide copy");

		cudaMalloc((void**)&(gpu_commons.gpu_guide_maxima), helper->maximaHelper.size() * sizeof(int));
		processor::cudaErrorCheck("maxima guide allocate");
		cudaMemcpy(gpu_commons.gpu_guide_maxima, helper->maximaHelper.data(), helper->maximaHelper.size() * sizeof(int), cudaMemcpyHostToDevice);
		processor::cudaErrorCheck("maxima guide copy");
	}
}


void processor::runLoop()
{
	auto go = true;

	auto doneCount = 0;
	
	while (go)
	{
		for (auto& feed : factory)
		{
			if (feed->rdy.load())
			{
				feed->contactProcessor();
			}
			else if (feed->isDone())
			{
				doneCount++;
			}

		}

		if (doneCount == feedThreads)
		{
			go = false;

		}

		doneCount = 0;

	}

}


void processor::run()
{
	std::vector<std::thread> threads;

#if runTiming
	clock_TCT();
#endif
	
	for (auto& feed : factory)
	{
		threads.emplace_back(std::thread(&feeder::run,feed.get()));
	}

	runLoop();

	for (auto& thread : threads)
	{
		thread.join();
	}

	std::vector<keyEntry> results;
	for (auto& feed : factory)
	{
		results.push_back(feed->getMax());
	}

	//deref the iterator returned from max
	maxDifference = *std::max_element(results.begin(), results.end());

	cudaDeviceReset();
	

#if runTiming
	record_TCT();
#endif

#if frameTiming
	for (auto& feed : factory)
	{
		feed->display_FPT();
	}
	display_FCT();
#endif // 

#if runTiming
	display_TCT();
#endif

}


//right now we are using two streams, one for each feeder, no reason it couldn't be a single stream since only one is ever in use at a time?
void processor::processFrame(std::vector<keyEntry>& input, const char feedID, cudaStream_t stream, gpuFeedMem feedMem)
{
	
#if debugOutput
	printData(0, input.data());
#endif // DEBUG	

#if frameTiming
	clock_FCT();
#endif // 0

	cudaMemcpyAsync(feedMem.gpu_permutation_data, input.data(), size_t(permutation_size) * size_t(rows) * sizeof(keyEntry), cudaMemcpyHostToDevice, stream);

	construct(feedMem.gpu_permutation_data, feedMem.gpu_matrix_UTM, gpu_commons.gpu_matrix_base, gpu_commons.gpu_guide_construction, permutation_size, matrix_size, rows, stream);

	summation(feedMem.gpu_matrix_UTM, feedMem.gpu_row_sums, gpu_commons.gpu_guide_summation, helper->summation_size, matrix_size, helper->summation_reductions, rows, helper->summation_threads, stream);

	maxima(feedMem.gpu_row_sums, feedMem.gpu_constant_maxima, gpu_commons.gpu_guide_maxima, helper->maxima_size , helper->maxima_reductions, helper->maxima_threads, rows, stream);

	//I would have thought this stream synchronize neccessary, but cutting it out doesn't seem to corrupt results and cuts total execution time in HALF.
	// It may just be lucky that streams finish before the prior stream does and cross frame contamination is avoided, and this makes me wonder if two sets of GPU memory arrays would help
	// to be fair, I'm not certain cross frame contamination ISN'T happening.
	//cudaStreamSynchronize(stream);


#if frameTiming
	record_FCT();
#endif // 0
}


const keyEntry processor::getMax()
{
	return maxDifference;
}

void processor::printCompletion(const std::string msg)
{
	printMtx.lock();
	std::cout << msg << std::endl;
	printMtx.unlock();
}

void processor::cudaErrorCheck(const std::string msg)
{
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaGetErrorString(err)<<" from "<<msg<< std::endl;
	}
}

const void processor::printData(const int frameNum, keyEntry * data)
{
	auto tableSize =  rows * permutation_size;
	printf("Data:\n");
	for (int i = 0; i < tableSize; i++)
	{
		printf("[%d] ", data[i]);
		if ((i + 1) % permutation_size == 0) printf("\n");
	}
}

void processor::createFeeds()
{
	//create a vector of all elements
	std::vector<keyEntry> activeElements;
	activeElements.resize(permutation_size);
	std::iota(activeElements.begin(), activeElements.end(), (keyEntry)1);

	// remove all elements in the preset
	for (keyEntry element : preset_permutation) activeElements.erase(std::remove(activeElements.begin(), activeElements.end(), element), activeElements.end());

	//for each non-preset element, create a new generator
	std::vector<std::unique_ptr<generator>> seedLine;
	for (keyEntry element : activeElements)
	{
		std::unique_ptr<generator> newGen = std::make_unique<generator>(preset_permutation, element, permutation_size);
		seedLine.push_back(std::move(newGen));
	}

	data_size = rows * permutation_size;
	
	for (auto it = factory.begin(); it != factory.end(); it++)
	{
		auto id = it - factory.begin();

		factory[id] = std::make_unique<feeder>(rows, permutation_size, this, id);
	}


	//assign generators to feeds
	for (auto it = seedLine.begin(); it != seedLine.end(); it++)
	{
		auto index = (it - seedLine.begin())%feedThreads;
		factory.at(index)->storeGenerator(std::move(*it));		
	}

	for (auto& feed : factory)
	{
		feed->loadGenerator();
	}

}

#if frameTiming
void processor::clock_FCT()
{
	fTic = std::chrono::high_resolution_clock::now();
}

void processor::record_FCT()
{
	std::chrono::duration<double, std::micro> stopwatch = std::chrono::high_resolution_clock::now() - fTic;
	frameCompTime.push_back(stopwatch.count());
}

void processor::display_FCT()
{
	auto size = (double)frameCompTime.size();
	auto sum = std::accumulate(frameCompTime.begin(), frameCompTime.end(), (double)0);
	auto max = std::max_element(frameCompTime.begin(), frameCompTime.end());
	auto min = std::min_element(frameCompTime.begin(), frameCompTime.end());

	std::cout << "Frame Computation Time (us): " << std::endl;
	std::cout << "Avg: " << std::to_string(sum / size) << std::endl;
	std::cout << "Min: " << std::to_string(*min) << std::endl;
	std::cout << "Max: " << std::to_string(*max) << std::endl;
}
#endif framTiming



#if runTiming
void processor::clock_TCT()
{
	tTic = std::chrono::high_resolution_clock::now();
}

void processor::record_TCT()
{
	timeElapsed = std::chrono::high_resolution_clock::now() - tTic;
}

const void processor::display_TCT()
{
	std::cout << "Total Completion Time: " << timeElapsed.count() << " seconds." << std::endl;
}
#endif
