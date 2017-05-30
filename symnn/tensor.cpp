#include "symnn/tensor.h"

namespace symnn {

void TensorTools::clip(Tensor& t, Tensor::REAL_TYPE left, Tensor::REAL_TYPE right)
{
    t.tvec() = t.tvec().cwiseMax(left).cwiseMin(right);
}

void TensorTools::constant(Tensor& t, Tensor::REAL_TYPE c)
{
    t.tvec() = t.tvec().constant(c);
}

void TensorTools::zero(Tensor& t)
{
    constant(t, 0.f);
}

void TensorTools::copy_elements(Tensor& target, const Tensor& source)
{
    memcpy(target.get_raw_ptr(), source.get_raw_ptr(), source.size() * sizeof(Tensor::REAL_TYPE));
}



} // end of namespace symnn