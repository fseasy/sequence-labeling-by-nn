#ifndef SYMNN_PARAM_NODES_H_INCLUDED_
#define SYMNN_PARAM_NODES_H_INCLUDED_

#include "symnn/symnn.h"
#include "symnn/model.h"
#include "symnn/nodes_macros.h"

namespace symnn {

class ParameterNodeBase : public Node
{
public:
    virtual void accumulate_grad(const Tensor& g) = 0;
};

class ParameterNode: public ParameterNodeBase
{
public:
    explicit ParameterNode(const Parameter& p);
    explicit ParameterNode(const LookupParameter& lp);
    void accumulate_grad(const Tensor& g) override;
    NODE_DEFINE_FLOW_IMPL()
private:
    Dim dim;
    Parameter params;
    LookupParameter lparams;
};

class ConstParameterNode : public Node
{
public:
    explicit ConstParameterNode(const Parameter& p);
    explicit ConstParameterNode(const LookupParameter& lp);
    NODE_DEFINE_FLOW_IMPL()
private:
    Dim dim;
    Parameter params;
    LookupParameter lparams;
};

class InputNode : public Node
{
public:
    InputNode(const Dim& d, const std::vector<real_t>& data);
    InputNode(const Dim& d, const std::vector<real_t>* pdata);
    NODE_DEFINE_FLOW_IMPL()
private:
    Dim dim;
    const std::vector<real_t> data;
    const std::vector<real_t>* pdata;
};

class ScalarInputNode : public Node
{
public:
    explicit ScalarInputNode(real_t s);
    explicit ScalarInputNode(const real_t* ps);
    NODE_DEFINE_FLOW_IMPL()
private:
    Dim dim;
    const real_t data;
    const real_t* pdata;
};


class LookupNode : public ParameterNodeBase
{
public:
    LookupNode(LookupParameter p, unsigned i);
    LookupNode(LookupParameter p, const unsigned* pi);
    LookupNode(LookupParameter p, const std::vector<unsigned>& indices);
    LookupNode(LookupParameter p, const std::vector<unsigned>* pindices);
    void accumulate_grad(const Tensor& g) override;
    NODE_DEFINE_FLOW_IMPL()
private:
    Dim dim;
    unsigned index;
    const unsigned* pindex;
    std::vector<unsigned> indices;
    LookupParameter params;
};

/***
 * inline implementation
 **/

ParameterNode::ParameterNode(const LookupParameter& lp)
    : dim(lp.get()->get_all_dimension()),
    lparams(lp) {}
ParameterNode::ParameterNode(const Parameter& p) 
    : dim(p.get()->get_dimension()), 
    params(p) {}


ConstParameterNode::ConstParameterNode(const Parameter& p) 
    : dim(p.get()->get_dimension()), 
    params(p) {}
ConstParameterNode::ConstParameterNode(const LookupParameter& lp) 
    : dim(lp.get()->get_all_dimension()), 
    lparams(lp) {}


InputNode::InputNode(const Dim& d, const std::vector<real_t>& data)
    :dim(d), data(data),
    pdata(&data) {}
InputNode::InputNode(const Dim& d, const std::vector<real_t>* pdata)
    : dim(d), data(),
    pdata(pdata) {}

ScalarInputNode::ScalarInputNode(real_t s)
    : data(s),
    pdata(&data) {}
ScalarInputNode::ScalarInputNode(const real_t* ps)
    : data(),
    pdata(ps) {}

LookupNode::LookupNode(LookupParameter p, unsigned i)
    : dim(p.get()->get_dimension()),
    index(i), pindex(&i), indices(),
    pindices(), params(p) {}
LookupNode::LookupNode(LookupParameter p, const unsigned* pi)
    : dim(p.get()->get_dimension()), index(),
    pindex(pi), indices(), pindices(),
    params(p) {}
LookupNode::LookupNode(LookupParameter p, const std::vector<unsigned>& indices)
    : dim(p.get()->get_dimension()),
    index(), pindex(), indices(indices),
    pindices(&this->indices),
    params(p) 
{
    dim.bd = indices.size();
}
LookupNode::LookupNode(LookupParameter p, const std::vector<unsigned>* pindices)
    : dim(p.get()->get_dimension()),
    index(), pindex(),
    indices(), pindices(pindices),
    params(p)
{
    dim.bd = pindices->size();
}


} // end of namespace symnn



#endif
