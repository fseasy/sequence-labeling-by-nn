#ifndef SYMNN_DIM_H_INCLUDED_
#define SYMNN_DIM_H_INCLUDED_

#include <vector>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <iostream>

namespace symnn {

class Dim {

public:
    Dim() {};
    Dim(std::initializer_list<unsigned> dim, unsigned batch_num=1U);

    std::size_t batch_size() const;
    std::size_t size() const { batch_size() * batch_num; }
    unsigned sum_dims() const;
    unsigned ndims() const { return static_cast<unsigned>(dim.size()); }
    unsigned nrows() const;
    unsigned ncols() const;

    void set_batch_num(unsigned new_batch_num) { batch_num = new_batch_num; };
    void resize(unsigned new_ndim);

    Dim single_batch() const;

    unsigned& operator[](unsigned i);
    const unsigned& operator[](unsigned i) const;
    const std::vector<unsigned>& get_dim() const { return dim; }
    unsigned get_batch_num() const { return batch_num; }
private:
    std::vector<unsigned> dim;
    unsigned batch_num; // we don't support batch, but it'll use in LookupNode
};

bool operator==(const Dim& lhs, const Dim& rhs);
std::ostream& operator<<(std::ostream& os, const Dim& d);

/**
 * inline implementation
 **/

inline
Dim::Dim(std::initializer_list<unsigned> dim, unsigned batch_num)
    :dim(dim), batch_num(batch_num)
{}

inline
std::size_t Dim::batch_size() const
{
    return std::accumulate(dim.begin(), dim.end(), 1U, std::multiplies<unsigned>());
}

inline
unsigned Dim::sum_dims() const
{
    return std::accumulate(dim.begin(), dim.end(), 0U);
}
inline
unsigned Dim::nrows() const
{
    return static_cast<unsigned>(dim.size()) > 0U ? dim[0] : 0U;
}

inline
unsigned Dim::ncols() const
{
    return static_cast<unsigned>(dim.size()) > 1U ? dim[1] : 1U; // 1 col
}



inline
void Dim::resize(unsigned new_ndim)
{
    dim.resize(new_ndim, 1U);
}

inline
Dim Dim::single_batch() const
{
    Dim d = *this;
    d.set_batch_num(1U);
    return d;
}

unsigned& Dim::operator[](unsigned i)
{
    return dim.at(i);
}

const unsigned& Dim::operator[](unsigned i) const
{
    return dim.at(i);
}

inline
bool operator==(const Dim& lhs, const Dim& rhs)
{
    return (lhs.get_dim() == rhs.get_dim()) && 
           (lhs.get_batch_num() == rhs.get_batch_num());
}

inline 
std::ostream& operator<<(std::ostream& os, const Dim& d)
{
    os << "dim[(";
    if (d.ndims() > 0U) { os << d[0]; }
    for (unsigned i = 1U; i < d.ndims(); ++i)
    {
        os << ", " << d[i];
    }
    os << ")";
    os << " x " << d.get_batch_num();
    os << "]";
}

} // end of namespace symnn


#endif