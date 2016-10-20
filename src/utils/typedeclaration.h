#ifndef TYPEDECLARATION_H_INCLUDED_
#define TYPEDECLARATION_H_INCLUDED_

#include <vector>
#include <array>
#include <utility>
#include <functional>
#include "dynet/dynet.h"

namespace slnn{
    using Index = int; // dynet::Dict return `int` as index 
    using IndexSeq = std::vector<Index>;
    using InstancePair = std::pair<IndexSeq, IndexSeq>;
    using Seq = std::vector<std::string>;
    template <int sz>
    using FeatureGroup = std::array<std::string, sz>;
    template <int sz>
    using FeatureGroupSeq = std::vector<FeatureGroup<sz>>;
    template <int sz>
    using FeatureIndexGroup = std::array<Index, sz>;
    template <int sz>
    using FeatureIndexGroupSeq = std::vector<FeatureIndexGroup<sz>>;

    template <int sz>
    using FeaturesIndex = std::array<Index, sz>;
    template <int sz>
    using FeaturesIndexSeq = std::vector<std::array<Index,sz>>;

    using NonLinearFunc = dynet::expr::Expression(const dynet::expr::Expression &); // an function : [input] -> expression , [output]-> expression 

namespace type{

using real = float;

} // end of namespace type
} // end of namespace slnn
#endif