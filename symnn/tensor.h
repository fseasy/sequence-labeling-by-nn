#ifndef SYMNN_TENSOR_H_INCLUDED_
#define SYMNN_TENSOR_H_INCLUDED_

#include "Eigen/Eigen"

#include "symnn/dim.h"
#include "symnn/type.h"

namespace symnn {

class Tensor {
public:


private:
    Dim d;
    real_t* v;

};


} // end of namespace symnn



#endif