#ifndef TYPEDECLARATION_INCLUDED_H
#define TYPEDECLARATION_INCLUDED_H
namespace slnn{
    using Index = int; // cnn::Dict return `int` as index 
    using IndexSeq = vector<Index>;
    using InstancePair = pair<IndexSeq, IndexSeq>;
}
#endif