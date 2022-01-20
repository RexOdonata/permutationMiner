#pragma once

#include <string>
#include <vector>
#include "defines.h"

#include <iostream>

class input
{
	public:
		int permutation_size;
		std::vector<keyEntry> preset_permutation;

		input(int argc, char* argv[]);

		void printAttributes();

};

