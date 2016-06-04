#ifndef TYPEDECLARATION_H_INCLUDED_
#define TYPEDECLARATION_H_INCLUDED_

#include <vector>
#include <initializer_list>
#include <utility> 
#include "cnn/cnn.h"

namespace slnn{
    using Index = int; // cnn::Dict return `int` as index 
    using IndexSeq = std::vector<Index>;
    using InstancePair = std::pair<IndexSeq, IndexSeq>;
    using Seq = std::vector<std::string>;
    using FeatureIndexSeq = std::vector<std::initializer_list<Index>>;

    using NonLinearFunc = cnn::expr::Expression(const cnn::expr::Expression &); // an function : [input] -> expression , [output]-> expression 
}
#endif