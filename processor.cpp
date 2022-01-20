#include "processor.h"

processor::processor(int set_permutation_Size, std::vector<keyEntry>& set_preset) :
	permutation_size(set_permutation_Size), matrix_size(set_permutation_Size* (set_permutation_Size - 1) / 2), preset_permutation(set_preset)
{
	createFeeds();

	helper = std::make_unique<guide>(permutation_size, matrix_size, rowsPerThread);

	initGPUMemory();

	gpuLock.store(0);
}

void processor::initGPUMemory()
{

	// allocate memory on the GPU
	{
		cudaMallocHost((void**)&gpu_permutation_data, size_t(rowsPerThread) * size_t(permutation_size) * sizeof(keyEntry));

		cudaMalloc((void**)&gpu_matrix_UTM, size_t(rowsPerThread) * size_t(matrix_size) * sizeof(keyEntry));

		cudaMalloc((void**)&gpu_row_sums, size_t(rowsPerThread) * sizeof(keyEntry));
	}

	// allocate and load in constants
	{
		cudaMalloc((void**)&gpu_constant_permutationSize, sizeof(int));
		cudaMemcpy(gpu_constant_permutationSize, &permutation_size, sizeof(permutation_size), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&gpu_constant_matrixSize, sizeof(int));
		cudaMemcpy(gpu_constant_matrixSize, &matrix_size, sizeof(matrix_size), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&gpu_constant_sumReductions, sizeof(int));
		cudaMemcpy(gpu_constant_sumReductions, &(helper->summation_reductions), sizeof(helper->summation_reductions), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&gpu_constant_maxReductions, sizeof(int));
		cudaMemcpy(gpu_constant_maxReductions, &(helper->maxima_reductions), sizeof(helper->maxima_reductions), cudaMemcpyHostToDevice);

		cudaMallocHost((void**)&gpu_constant_maxima, sizeof(keyEntry));
	}

	// create the base matrix and load it
	{
		std::vector<keyEntry> base(matrix_size);
		auto start = 0;
		for (int i = permutation_size - 1; i > 0; i--)
		{
			std::iota(base.begin() + start, base.begin() + start + i, 1);
			start += i;
		}

		cudaMalloc((void**)&gpu_matrix_base, matrix_size * sizeof(keyEntry));
		cudaMemcpy(gpu_matrix_base, base.data(), matrix_size * sizeof(keyEntry), cudaMemcpyHostToDevice);
	}

	// allocate and load helpers
	{
		cudaMalloc((void**)&gpu_guide_construction, helper->constructionHelper.size() * sizeof(matrixIndexPair));
		cudaMemcpy(gpu_guide_construction, helper->constructionHelper.data(), helper->constructionHelper.size() * sizeof(matrixIndexPair), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&gpu_guide_summation, helper->summationHelper.size() * sizeof(reductionGuide));
		cudaMemcpy(gpu_guide_summation, helper->summationHelper.data(), helper->summationHelper.size() * sizeof(reductionGuide), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&gpu_guide_maxima, helper->maximaHelper.size() * sizeof(reductionGuide));
		cudaMemcpy(gpu_guide_maxima, helper->maximaHelper.data(), helper->maximaHelper.size() * sizeof(reductionGuide), cudaMemcpyHostToDevice);
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

	for (auto& thread : threads)
	{
		thread.join();
	}

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

void processor::contactProcessor(std::vector<keyEntry>& input, char feedID)
{
	while (feedID == gpuLock.load())
	{

	}

#if debugOutput
	printData(0, input.data());
#endif // DEBUG	

	auto result = processDataFrame(input);
	maxDifference = std::max(maxDifference, result);

	gpuLock.store(feedID+lockBreaker);
}

void processor::relaxLock()
{
	lockBreaker = 1;
}

keyEntry processor::getMax()
{
	return maxDifference;
}

void processor::printData(int frameNum, keyEntry * data)
{
	auto tableSize =  rowsPerThread * permutation_size;
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


	rowsPerThread = (CORES / matrix_size);

	data_size = rowsPerThread * permutation_size;
	

	factory[0] = std::make_unique<feeder>(rowsPerThread, permutation_size, this, 0);
	factory[1] = std::make_unique<feeder>(rowsPerThread, permutation_size, this, 1);



	//assign generators to feeds
	for (auto it = seedLine.begin(); it != seedLine.end(); it++)
	{
		auto index = (it - seedLine.begin())%2;
		factory.at(index)->storeGenerator(std::move(*it));		
	}

	//do initial source loading
	factory[0]->loadGenerator();
	factory[1]->loadGenerator();

}

keyEntry processor::processDataFrame(std::vector<keyEntry>& input)
{
#if frameTiming
	clock_FCT();
#endif // 0

	keyEntry result = 0;

	cudaMemcpy(gpu_permutation_data, input.data(), size_t(permutation_size) * size_t(rowsPerThread) * sizeof(keyEntry), cudaMemcpyHostToDevice);

	construct(gpu_permutation_data, gpu_matrix_UTM, gpu_guide_construction, gpu_constant_permutationSize, matrix_size, rowsPerThread);

	difference(gpu_matrix_UTM, gpu_matrix_base, matrix_size, rowsPerThread);

	summation(gpu_matrix_UTM, gpu_row_sums, gpu_guide_summation, gpu_constant_matrixSize, gpu_constant_sumReductions, rowsPerThread, helper->summation_threads);

	maxima(gpu_row_sums, gpu_guide_maxima, gpu_constant_matrixSize, gpu_constant_maxima, gpu_constant_maxReductions, helper->maxima_threads);

	cudaMemcpy(&result, gpu_constant_maxima, sizeof(keyEntry), cudaMemcpyDeviceToHost);

#if frameTiming
	record_FCT();
#endif // 0

	return result;

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

void processor::display_TCT()
{
	std::cout << "Total Completion Time: " << timeElapsed.count() << " seconds." << std::endl;
}
#endif
