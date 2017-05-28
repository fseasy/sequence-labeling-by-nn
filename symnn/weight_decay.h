#ifndef SYMNN_WEIGHT_DECAY_H_INCLUDED_
#define SYMNN_WEIGHT_DECAY_H_INCLUDED_

#include <stdexcept>
#include <cmath>
#include <iostream>

#include "symnn/type.h"
namespace symnn {

struct L2WeightDecay
{
public:
    explicit L2WeightDecay(real_t lambda = 1e-6f);
    void set_lambda(real_t lambda);
    void update_weight_decay(unsigned num_udpates = 1);
    real_t current_weight_decay() const { return weight_decay; };
    bool parameters_need_rescaled() const;
    void reset_weight_decay();
private:
    real_t weight_decay;
    real_t lambda;
};

/**
 * inline implementaiton
 *****/

inline
L2WeightDecay::L2WeightDecay(real_t lambda)
    :weight_decay(1)
{
    set_lambda(lambda);
}

inline
void L2WeightDecay::set_lambda(real_t lambda)
{
    if (lambda < 0)
    {
        throw std::domain_error("lambda value less than 0");
    }
    this->lambda = lambda;
}

inline
void L2WeightDecay::update_weight_decay(unsigned num_updates)
{
    if (num_updates == 0) return;
    else if (num_updates == 1)
    {
        weight_decay -= weight_decay * lambda;
    }
    else
    {
        weight_decay = weight_decay * std::pow(1 - lambda, num_updates);
    }
}

inline
bool L2WeightDecay::parameters_need_rescaled() const
{
    return (weight_decay < 0.25f);
}

inline
void L2WeightDecay::reset_weight_decay()
{
    std::cerr << "RESCALE WEIGHT DECAY FROM "
        << weight_decay
        << " to 1.0\n";
    weight_decay = 1.0f;
}


} // end of namespace symnn


#endif