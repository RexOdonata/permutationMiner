#include "input.h"

input::input(int argc, char* argv[])
{
	bool getInt = false;
	char mode;

	permutation_size = 0;

	for (int i = 1; i < argc; i++)
	{

		std::string argument(argv[i]);

		if (getInt)
		{
			switch (mode)
			{
			case 'n':
			{
				permutation_size = std::stoi(argument, nullptr);
				getInt = false;
				break;
			}
			case 'p':
			{
				keyEntry numberIn = std::stoi(argument, nullptr);
				if (numberIn > 0)
				{
					preset_permutation.push_back(numberIn);
				}
				else
				{
					getInt = false;
				}
				break;
			}
			}
		}
		else
		{
			if (argument.std::string::compare("-N") == 0)
			{
				getInt = true;
				mode = 'n';
			}
			else if (argument.std::string::compare("-P") == 0)
			{
				getInt = true;
				mode = 'p';
			}
		}

	}
}

void input::printAttributes()
{
	std::cout << "N=" << permutation_size << std::endl;
	std::cout << "Preset:" << std::endl;
	for (auto& preset : preset_permutation)
	{
		std::cout << " " << preset << " " << std::endl;
	}
}
