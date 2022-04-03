#pragma once
#include "defines.h"
#include <algorithm>
#include <vector>
#include <numeric>

#include <string>

enum class generatorState
{
	WAITING,
	ACTIVE,
	BOUNCE,
	DONE,
	ERROR
};

class generator
{
	private:
		std::vector<int> permutationClock;
		int primary_index;
		int secondary_index;

		const int permutation_size;
		const int seedPermutation_size;

		std::vector<keyEntry> active_permutation;
		std::vector<keyEntry> seed_permutation;

	public:
		generatorState state;

		generator(std::vector<keyEntry> set_preset, keyEntry seedElement, int permutation_size);

		void advancePermutation();

		bool advanceToNext();

		void zeroPad();

		const void loadData(std::vector<keyEntry>& data,int address);

		const void printPermutation();

		const std::string getLabel();

#ifdef timingOutput

#endif // timing


		
};

