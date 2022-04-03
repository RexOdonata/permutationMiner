#include "generator.h"

const std::string generator::getLabel()
{
	std::string label = "";

	for (auto element : seed_permutation)
	{
		label += std::to_string(element) + " ";
	}
	return label;
}

generator::generator(std::vector<keyEntry> set_preset, keyEntry seedElement, int set_permutation_size) : permutation_size(set_permutation_size), seedPermutation_size(set_preset.size()+1)
{
	seed_permutation = set_preset;
	seed_permutation.push_back(seedElement);

	primary_index = 1;
	secondary_index = 0;
		
	//init active permutation
	active_permutation.resize(permutation_size,0);

	std::iota(active_permutation.begin(), active_permutation.end(), (keyEntry)1);

	for (keyEntry element : seed_permutation)
	{
		active_permutation.erase(std::remove(active_permutation.begin(), active_permutation.end(), element), active_permutation.end());
	}

	//init clock to 0s
	permutationClock.resize(active_permutation.size(), 0);	

	state = generatorState::WAITING;
}

// creates a base permutation
void generator::zeroPad()
{
	std::iota(seed_permutation.begin(), seed_permutation.end(),1);
	std::iota(active_permutation.begin(), active_permutation.end(), seed_permutation.back()+1);
}

void generator::advancePermutation()
{
	
	if (permutationClock.at(primary_index) < primary_index)
	{
		if (primary_index % 2 != 0) secondary_index = permutationClock.at(primary_index);
		else secondary_index = 0;

		std::swap(active_permutation.at(primary_index), active_permutation.at(secondary_index));

		permutationClock.at(primary_index)++;

		primary_index = 1;
		//iteration succesful
		state = generatorState::ACTIVE;
	}
	else if (permutationClock.at(primary_index) == primary_index)
	{
		permutationClock.at(primary_index) = 0;
		primary_index++;
		if (primary_index == active_permutation.size()) state = generatorState::DONE;
		else											state = generatorState::BOUNCE;
	}
	else generatorState::ERROR;
	
}

bool generator::advanceToNext()
{

	while (1)
	{
		if (state == generatorState::ACTIVE) return false;
		else if (state == generatorState::BOUNCE) advancePermutation();
		else if (state == generatorState::DONE) return true;
	}
}


const void generator::loadData(std::vector<keyEntry>& data, int row)
{
	int startAddress = row * permutation_size;
	std::copy(seed_permutation.begin(), seed_permutation.end(), data.begin() + startAddress);
	std::copy(active_permutation.begin(), active_permutation.end(), data.begin() + startAddress + seedPermutation_size);
}

const void generator::printPermutation()
{
	for (keyEntry element : seed_permutation) printf("[%d] ",element);
	for (keyEntry element : active_permutation) printf("[%d] ", element);
	printf("\n");

}