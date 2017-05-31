#ifndef SYMNN_GLOBAL_VAR_H_INCLUDED_
#define SYMNN_GLOBAL_VAR_H_INCLUDED_

#include <random>

namespace symnn {

class Device;


extern Device* global_device;
extern std::mt19937* global_rng;

}


#endif