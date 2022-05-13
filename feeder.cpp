#include "feeder.h"

feeder::feeder(int set_rowsPerThread, int set_permutation_size, processor * set_parent, char set_id) :
	rows(set_rowsPerThread), permutation_size(set_permutation_size), proc(set_parent), id(set_id), matrix_size((permutation_size-1)*(permutation_size)/2)
{
	data.resize(static_cast<long long>(rows) * static_cast<long long>(permutation_size), 0);

	cudaStreamCreate(&stream);

	// allocate memory on the GPU
	{
		cudaMallocHost((void**)&(gpu_feed_private.gpu_permutation_data), size_t(rows) * size_t(permutation_size) * sizeof(keyEntry));
		processor::cudaErrorCheck("Allocate feeder Permution Array");

		cudaMalloc((void**)&(gpu_feed_private.gpu_matrix_UTM), size_t(rows) * size_t(matrix_size) * sizeof(keyEntry));
		processor::cudaErrorCheck("Allocate feeder UTM matrix array");

		cudaMalloc((void**)&(gpu_feed_private.gpu_row_sums), size_t(rows) * sizeof(keyEntry));
		processor::cudaErrorCheck("Allocate row sums array");

		cudaMallocHost((void**)&(gpu_feed_private.gpu_constant_maxima), sizeof(keyEntry));
		processor::cudaErrorCheck("Allocate maxima data element");

		const keyEntry zero = 0;
		cudaMemcpy(gpu_feed_private.gpu_constant_maxima, &zero, sizeof(keyEntry), cudaMemcpyHostToDevice);
		processor::cudaErrorCheck("copy initialize maxima to 0");
	}

	max = 0;
}

void feeder::storeGenerator(std::unique_ptr<generator> in)
{
	seeds.push_back(std::move(in));
}

void feeder::loadGenerator()
{
	source = std::move(seeds.back());
	seeds.pop_back();
	source->state = generatorState::ACTIVE;
}

void feeder::run()
{
	auto procComplete = false;
	while (!procComplete)
	{
#if frameTiming
		clock_FPT();
#endif // DEBUG

		procComplete = fillFrame();

#if frameTiming
		record_FPT();
#endif // DEBUG

		rdy.store(true);

		spinLock();
	}

	done = true;
	
	//copy result into private
	cudaStreamSynchronize(stream);
	cudaMemcpy(&max, gpu_feed_private.gpu_constant_maxima, sizeof(keyEntry), cudaMemcpyDeviceToHost);

}

void feeder::contactProcessor()
{
	proc->processFrame(data, id, stream, gpu_feed_private);
	rdy.store(false);
}

const keyEntry feeder::getMax()
{
	return max;
}

bool feeder::isDone()
{
	return done;
}

bool feeder::fillFrame()
{
	for (int row = 0; row < rows; row++)
	{
		auto needNewGen = source->advanceToNext();

		if (needNewGen)
		{
			std::string label = source->getLabel();
			proc->printCompletion("Seed " + label + "is done.\n");
			if (seeds.size() > 0) loadGenerator();
			else
			{
				padFrame(row);
				return true;
			}
		}

		source->loadData(data, row);
		source->advancePermutation();

	}
	return false;
}

void feeder::padFrame(int startRow)
{
	source->zeroPad();
	for (int row = startRow; row < rows; row++)
	{
		source->loadData(data,row);
	}
}

void feeder::spinLock()
{
	while (rdy.load() == true)
	{
		//busy wait
	}
}

#if frameTiming

	void feeder::clock_FPT()
	{
		fTic = std::chrono::high_resolution_clock::now();
	}
	
	void feeder::record_FPT()
	{
		std::chrono::duration<double, std::micro> stopwatch = std::chrono::high_resolution_clock::now() - fTic;
		frameFillTime.push_back(stopwatch.count());
	}
	
	void feeder::display_FPT()
	{
		auto size = (double)frameFillTime.size();
		auto sum = std::accumulate(frameFillTime.begin(), frameFillTime.end(), (double)0);
		auto max = std::max_element(frameFillTime.begin(), frameFillTime.end());
		auto min = std::min_element(frameFillTime.begin(), frameFillTime.end());
	
		std::cout << "Frame Generation Time (us): " << std::endl;
		std::cout << "Avg: " << std::to_string(sum / size) << std::endl;
		std::cout << "Min: " << std::to_string(*min) << std::endl;
		std::cout << "Max: " << std::to_string(*max) << std::endl;
	}
	
#endif // frameTiming