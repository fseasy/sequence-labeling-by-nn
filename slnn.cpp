#include <iostream>

#include "symnn/symnn.h"

int main(int argc, char* argv[])
{
    std::cout << "start my graduation coding!\n"
              << "I'll start by coping dynet!\n";
    for (int i = 0; i < argc; ++i)
    {
        std::cout << argv[i] << std::endl;
    }
    system("pause");
    return 0;
}