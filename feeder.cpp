#include "feeder.h"

feeder::feeder(int set_rowsPerThread, int set_permutation_size, processor * set_parent, char set_id) :
	rowsToFill(set_rowsPerThread), permutation_size(set_permutation_size), proc(set_parent), id(set_id)
{
	data.resize(static_cast<long long>(rowsToFill) * static_cast<long long>(permutation_size), 0);
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
	while (done == false)
	{
#if frameTiming
		clock_FPT();
#endif // DEBUG

		done=fillFrame();

#if frameTiming
		record_FPT();
#endif // DEBUG

		proc->contactProcessor(data,id);
	}

	proc->relaxLock();

}

bool feeder::fillFrame()
{
	for (int row = 0; row < rowsToFill; row++)
	{
		auto needNewGen = source->advanceToNext();

		if (needNewGen)
		{
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
	for (int row = startRow; row < rowsToFill; row++)
	{
		source->loadData(data,row);
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