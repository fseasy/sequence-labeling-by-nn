#ifndef TYPEDECLARATION_H_INCLUDED_
#define TYPEDECLARATION_H_INCLUDED_

namespace slnn{
    using Index = int; // cnn::Dict return `int` as index 
    using IndexSeq = std::vector<Index>;
    using InstancePair = std::pair<IndexSeq, IndexSeq>;
    using Seq = std::vector<std::string>;
}
#endif