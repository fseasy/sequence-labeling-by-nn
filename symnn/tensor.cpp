#include <random>

#include "symnn/tensor.h"
#include "symnn/global_vars.h"

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

void TensorTools::identity(Tensor& t)
{
    SLNN_ARG_CHECK(t.get_dim().ndims() == 2U && t.get_dim().nrows() == t.get_dim().ncols(),
                   MODULE_SYMNN_NAME,
                   "TensorTools::identity failed "
    "for un-equal row and col");
    std::size_t pos = 0U,
        nrow = t.get_dim().nrows();
    for (std::size_t i = 0U; i < nrow; ++i) {
        for (std::size_t j = 0U; j < nrow; ++j)
        {
            t.get_raw_ptr()[pos++] = (i == j? 1.f : 0.f);
        }
    }
}

void TensorTools::copy_elements(Tensor& target, const Tensor& source)
{
    memcpy(target.get_raw_ptr(), source.get_raw_ptr(), source.size() * sizeof(Tensor::REAL_TYPE));
}

void TensorTools::randomize_normal(Tensor& val,
                                   Tensor::REAL_TYPE mean,
                                   Tensor::REAL_TYPE stddev)
{
    std::normal_distribution<Tensor::REAL_TYPE> distribution(mean, stddev);
    auto generate1 = [&] { return distribution(*global_rng); };
    std::generate(val.get_raw_ptr(), val.get_raw_ptr() + val.size(),
                  generate1);
}

void TensorTools::randomize_uniform(Tensor& val,
                                    Tensor::REAL_TYPE left,
                                    Tensor::REAL_TYPE right)
{
    std::uniform_real_distribution<Tensor::REAL_TYPE> distribution(left, right);
    auto generate1 = [&] { return distribution(*global_rng); };
    std::generate(val.get_raw_ptr(), val.get_raw_ptr() + val.size(),
                  generate1);
}

} // end of namespace symnn