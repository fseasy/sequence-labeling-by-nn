#ifndef TYPEDECLARATION_H_INCLUDED_
#define TYPEDECLARATION_H_INCLUDED_

#include <vector>
#include <array>
#include <utility> 
#include "cnn/cnn.h"

namespace slnn{
    using Index = int; // cnn::Dict return `int` as index 
    using IndexSeq = std::vector<Index>;
    using InstancePair = std::pair<IndexSeq, IndexSeq>;
    using Seq = std::vector<std::string>;
    template <int sz>
    using FeaturesIndex = std::array<Index, sz>;
    template <int sz>
    using FeaturesIndexSeq = std::vector<std::array<Index,sz>>;

    using NonLinearFunc = cnn::expr::Expression(const cnn::expr::Expression &); // an function : [input] -> expression , [output]-> expression 
}
#endif