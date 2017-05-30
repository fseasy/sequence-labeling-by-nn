#ifndef SYMNN_TENSOR_H_INCLUDED_
#define SYMNN_TENSOR_H_INCLUDED_

#include "Eigen/Eigen"
#include "unsupported/Eigen/CXX11/Tensor"

#include "symnn/dim.h"
#include "symnn/type.h"
#include "symnn/device.h"

namespace symnn {

class Tensor {
    
public:
    using REAL_TYPE = real_t;
    using EIGEN_MATRIX_TYPE = Eigen::MatrixXf;
    using EIGEN_VECTOR_TYPE = Eigen::VectorXf;
    template <unsigned N>
    using EIGEN_TENSOR_N_TYPE = Eigen::Tensor<REAL_TYPE, N>;

    /**
     * Eigen::Map is Eigen's interface with raw buffers
     * it is the key to self-management-memory-pool
     */
    using MATRIX_MAP = Eigen::Map<EIGEN_MATRIX_TYPE>;
    using VECTOR_MAP = Eigen::Map<EIGEN_VECTOR_TYPE>;
    template <unsigned N>
    using TENSOR_N_MAP = Eigen::TensorMap<EIGEN_TENSOR_N_TYPE<N>>;
public:
    Tensor();
    Tensor(const Dim& d, REAL_TYPE* v, MemPoolType mpt, Device* dev);
    std::size_t size() const { return d.size(); }
    void set_raw_ptr(REAL_TYPE* ptr) { v = ptr; }
    void set_mpt(const MemPoolType& mpt_) { mpt = mpt_; }

    MATRIX_MAP operator*();
    const MATRIX_MAP operator*() const;
    
    VECTOR_MAP vec();
    const VECTOR_MAP vec() const;
    
    TENSOR_N_MAP<1U> tvec();
    const TENSOR_N_MAP<1U> tvec() const;
    
    TENSOR_N_MAP<2U> tbvec();
    const TENSOR_N_MAP<2U> tbvec() const;
    
    template<unsigned Order>
    TENSOR_N_MAP<Order> t();
    template<unsigned Order>
    const TENSOR_N_MAP<Order> t() const;

    template<unsigned Order>
    TENSOR_N_MAP<Order+1> tb();
    template<unsigned Order>
    const TENSOR_N_MAP<Order+1> tb() const;

public:
    REAL_TYPE* get_raw_ptr() { return v; }
    const REAL_TYPE* get_raw_ptr() const { return v; }
    const Dim& get_dim() const { return d; }
    Dim& get_dim() { return d; }
    const MemPoolType& get_mpt() const { return mpt; }
    const Device* get_device() const { return dev; }
    Device*& get_device() { return dev; }
private:
    Dim d;
    REAL_TYPE* v;
    MemPoolType mpt;
    Device* dev;
};

Tensor::REAL_TYPE as_scalar(const Tensor& t);
std::vector<Tensor::REAL_TYPE> as_vector(const Tensor& v);


struct TensorTools
{
    static void clip(Tensor& d, Tensor::REAL_TYPE left, Tensor::REAL_TYPE right);

    static void constant(Tensor& d, Tensor::REAL_TYPE c);

    static void zero(Tensor& d);
    
    static void copy_elements(Tensor& target, const Tensor& source);

};

/**
 * inline/template implementation
 ***/

inline
Tensor::Tensor()
    :d(Dim()),
    v(nullptr), mpt(MemPoolType::NONE),
    dev(nullptr) {}

inline
Tensor::Tensor(const Dim& d, REAL_TYPE* v, MemPoolType mpt, Device* dev_)
    : d(d), v(v), mpt(mpt),
    dev(dev_){}

inline
Tensor::MATRIX_MAP Tensor::operator*()
{
    return static_cast<const Tensor*>(this)->operator*();
}

inline
const Tensor::MATRIX_MAP Tensor::operator*() const
{
    return MATRIX_MAP(v, d.nrows(), d.ncols());
}

inline
Tensor::VECTOR_MAP Tensor::vec()
{
    return static_cast<const Tensor*>(this)->vec();
}

inline
const Tensor::VECTOR_MAP Tensor::vec() const
{
    return VECTOR_MAP(v, d.size());
}

inline
Tensor::TENSOR_N_MAP<1U> Tensor::tvec()
{
    return static_cast<const Tensor*>(this)->tvec();
}

inline
const Tensor::TENSOR_N_MAP<1U> Tensor::tvec() const
{
    return TENSOR_N_MAP<1U>(v, d.size());
}

inline
Tensor::TENSOR_N_MAP<2U> Tensor::tbvec()
{
    return static_cast<const Tensor*>(this)->tbvec();
}

inline
const Tensor::TENSOR_N_MAP<2U> Tensor::tbvec() const
{
    return TENSOR_N_MAP<2U>(v, d.batch_size(),
                            d.get_batch_num());
}





} // end of namespace symnn



#endif