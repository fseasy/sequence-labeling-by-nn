#ifndef TYPEDECLARATION_H_INCLUDED_
#define TYPEDECLARATION_H_INCLUDED_

#include <vector>
#include <utility> 
#include "cnn/cnn.h"

namespace slnn{
    using Index = int; // cnn::Dict return `int` as index 
    using IndexSeq = std::vector<Index>;
    using InstancePair = std::pair<IndexSeq, IndexSeq>;
    using Seq = std::vector<std::string>;

    using NonLinearFunc = cnn::expr::Expression(const cnn::expr::Expression &);
}
#endif