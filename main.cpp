#include <iostream>
#include <string>

#include "input.h"
#include "processor.h"



int main(int argc, char* argv[])
{
	input arguments(argc, argv);
	arguments.printAttributes();
	processor miner(arguments.permutation_size, std::move(arguments.preset_permutation));
	miner.run();
	std::cout << "Result: " << miner.getMax() << std::endl;

	return 0;
}