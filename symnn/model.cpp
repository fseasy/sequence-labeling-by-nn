#include "symnn/model.h"
#include "symnn/device.h"
#include "symnn/global_vars.h"
using namespace std;

namespace symnn {

ParameterStorage::ParameterStorage(const Dim& dim_, real_t minmax)
    :dim(dim_),
    value(dim_,
          global_device->allocate_space4tensor(MemPoolType::PS,
                                              dim_),
          MemPoolType::PS,
          global_device),
    g(dim_,
      global_device->allocate_space4tensor(MemPoolType::PS),
      MemPoolType::PS,
      global_device)
{
    TensorTools::zero(g);
    if (std::abs(minmax - 0.f) <= 1e-6f)
    {
        ParameterInitGlorot().initialize_params(value);
    }
    else
    {
        ParameterInitUniform(minmax).initialize_params(value);
    }
}

ParameterStorage::ParameterStorage(const Dim& d, const ParameterInit& init)
    :dim(dim_),
    value(dim_,
          global_device->allocate_space4tensor(MemPoolType::PS,
                                               dim_),
          MemPoolType::PS,
          global_device),
    g(dim_,
      global_device->allocate_space4tensor(MemPoolType::PS),
      MemPoolType::PS,
      global_device)
{
    TensorTools::zero(g);
    init.initialize_params(value);
}

size_t ParameterStorage::size() const
{
    return dim.size();
}

void ParameterStorage::zero()
{
    TensorTools::zero(value);
    clear();
}

void ParameterStorage::copy(const ParameterStorage& other)
{
    SLNN_ARG_CHECK(other.get_dimension() == dim,
                   MODULE_SYMNN_NAME,
                   "ParameterStorage::copy failed "
                   "for not equal dimention");
    TensorTools::copy_elements(value, other.get_value());
}

void ParameterStorage::clear() {
    if (g.get_raw_ptr() != nullptr)
    {
        TensorTools::zero(g);
    }
}

void ParameterStorage::clip(real_t left, real_t right)
{
    TensorTools::clip(value, left, right);
}

LookupParameterStorage::LookupParameterStorage(unsigned n, const Dim& d,
                                               const ParameterInit& init)
    :dim(d),
    all_updated(false)
{
    all_dim = dim;
    // add the num of Parameter to dim
    all_dim.get_dim().push_back(n);
    all_grads.get_dim() = all_values.get_dim() = all_dim;
    all_grads.get_device() = all_values.get_device() = global_device;
    global_device->allocate_tensor(MemPoolType::PS,
                                   all_values);
    global_device->allocate_tensor(MemPoolType::PS,
                                   all_grads);
    init.initialize_params(all_values);
    initialize_lookups();
}

void LookupParameterStorage::initialize_lookups()
{
    int nr_param = all_dim.get_dim().back();
    unsigned dim_sz = dim.size();
    if (values.size() == 0U)
    {
        values.resize(nr_param);
        for (unsigned i = 0; i < nr_param; ++i)
        {
            // can this make it not aligned??
            values[i] = Tensor(dim,
                               all_values.get_raw_ptr() + i * dim_sz,
                               all_values.get_mpt(),
                               all_values.get_device());
        }
    }
    if (grads.size() == 0U && all_grads.get_raw_ptr() != nullptr)
    {
        grads.resize(nr_param);
        for (unsigned i = 0; i < nr_param; ++i)
        {
            grads[i] = Tensor(dim, 
                              all_grads.get_raw_ptr + i*dim_sz,
                              all_grads.get_mpt(),
                              all_grads.get_device());
        }
    }
}

void LookupParameterStorage::zero()
{
    TensorTools::zero(all_values);
}

size_t LookupParameterStorage::size() const
{
    return all_dim.size();
}

void LookupParameterStorage::copy(const LookupParameterStorage& other)
{
    SLNN_ARG_CHECK(all_dim != other.get_all_dimension(),
                   MODULE_SYMNN_NAME,
                   "LookupParameterStorage::copy failed "
                   "for not equal dim");
    TensorTools::copy_elements(all_values, other.get_all_values());
}

void LookupParameterStorage::clear()
{
    if (all_updated)
    {
        TensorTools::zero(all_grads);
    }
    else
    {
        for (unsigned i: non_zero_grads) 
        {
            TensorTools::zero(grads[i]);
        }
    }
    non_zero_grads.clear();
    all_updated = false;
}




} // end of namespace symnn