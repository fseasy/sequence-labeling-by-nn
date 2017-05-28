/*
* SymNN - Symbolical Neural Network framwork
* copying from [dynet](https://github.com/clab/dynet)

--------------------------------------------------------------------
 
Copyright 2015 Chris Dyer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SYMNN_SYMNN_H_INCLUEDD_
#define SYMNN_SYMNN_H_INCLUDED_

#include <initializer_list>
#include <vector>

#include "symnn/type.h"
#include "symnn/parameters.h"
#include "symnn/tensor.h"
#include "symnn/dim.h"
#include "symnn/param_nodes.h"
#include "symnn/exec.h"

namespace symnn{

// pre-declaring.
// for decoupling the dependency.
class Node;
namespace expr { class Expression; }

class ComputationGraph
{
    friend class expr::Expression;
public:
    // constructor / deconstructor
    ComputationGraph();
    ~ComputationGraph();

    // flow
    const Tensor& forward(const expr::Expression& last);
    void backward(const expr::Expression& last);

    const Dim& get_dimention(node_id_t i) const;

    void clear();

protected:
    /**
     * interface to inner.
     **/

    const Tensor& forward(node_id_t i);
    void backward(const node_id_t i);

    // inputs
    node_id_t add_input(real_t i);
    node_id_t add_input(real_t* pi);
    node_id_t add_input(const Dim& d, const std::vector<real_t>& v);
    node_id_t add_input(const Dim& d, const std::vector<real_t>* pv);

    // parameters
    node_id_t add_parameter(const Parameter& p);
    node_id_t add_parameter(const LookupParameter& lp);

    node_id_t add_const_parameter(const Parameter& p);
    node_id_t add_const_parameter(const LookupParameter& lp);

    node_id_t add_lookup(const LookupParameter& l, unsigned index);
    node_id_t add_lookup(const LookupParameter& l, const unsigned* pindex);
    node_id_t add_lookup(const LookupParameter& l, const std::vector<unsigned>& indices);
    node_id_t add_lookup(const LookupParameter& l, const std::vector<unsigned>* pindices);

    node_id_t add_const_lookup(const LookupParameter& l, unsigned index);
    node_id_t add_const_lookup(const LookupParameter& l, const unsigned* pindex);
    node_id_t add_const_lookup(const LookupParameter& l, const std::vector<unsigned>& indices);
    node_id_t add_const_lookup(const LookupParameter& l, const std::vector<unsigned>* pindices);
    // functions
    template<typename Function>
    node_id_t add_function(
        const std::initializer_list<node_id_t>& args);

    template<typename Function, typename T>
    node_id_t add_function(const T& args);

    template<typename Function, typename... Args>
    node_id_t add_function(const std::initializer_list<node_id_t>& args,
                           Args&&... side_information);

    template<typename Function, typename T, typename... Args>
    node_id_t add_function(const T& args, Args&&... side_infomation);

private:
    template<typename NodeT>
    node_id_t do_add_node(NodeT&& node);

    void set_dim_cg4new_node(node_id_t i);

private:
    std::vector<Node*> nodes; // in topological order
    std::vector<node_id_t> parameter_nodes; // the parameters nodes (can be update)
    ExecutionEngine *engine;
};

class Node
{
    friend class ComputationGraph;
public:
    virtual ~Node() {};
    
    virtual const Dim& dim_forward(const std::vector<const Dim*>& xs) const = 0;

    virtual void forward_impl(const std::vector<const Tensor*>& xs,
                              Tensor& fx) const = 0;

    virtual void backward_impl(const std::vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const = 0;
    virtual void forward(const std::vector<const Tensor*>& xs,
                         Tensor& fx) const final;
    virtual void backward(const std::vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const final;

    const Dim& get_dimention() const { return dim; }
    std::size_t arity() const { return args.size(); }
protected:
    Node() : args() {}
    explicit Node(std::initializer_list<node_id_t>& args) : args(args) {}
    template<typename T>
    Node(const T& args) : args(args.begin(), args.end()) {}
protected:
    std::vector<node_id_t> args; 
    Dim dim;
    ComputationGraph *pcg;
};


/**************
 * template/inlie implementation
 ********/

inline
ComputationGraph::ComputationGraph()
{
    engine = new SimpleExcutionEngine(this);
}

inline
const Tensor& ComputationGraph::forward(node_id_t i)
{
    return engine->forward(i);
}
inline
void ComputationGraph::backward(node_id_t i)
{
    return engine->backward(i);
}


template<typename NodeT>
node_id_t ComputationGraph::do_add_node(NodeT&& node)
{
    // about forward: http://en.cppreference.com/w/cpp/utility/forward
    // as alternative, we can use MACROS to implement this.
    // #define DO_ADD_NODE(NODE) \ ....
    node_id_t new_node_id = nodes.size();
    nodes.push_back(std::forward<NodeT>(node));
    set_dim_cg4new_node(new_node_id);
    return new_node_id;
}

inline
node_id_t ComputationGraph::add_input(real_t v)
{
    return do_add_node(new ScalarInputNode(v));
}

inline
node_id_t ComputationGraph::add_input(real_t* pv)
{
    return do_add_node(new ScalarInputNode(pv));
}

inline
node_id_t ComputationGraph::add_input(const Dim& d, const std::vector<real_t>& v)
{
    return do_add_node(new InputNode(d, v));
}

inline
node_id_t ComputationGraph::add_input(const Dim& d, const std::vector<real_t>* pv)
{
    return do_add_node(new InputNode(d, pv));
}

inline
node_id_t ComputationGraph::add_parameter(const Parameter& p)
{
    node_id_t new_node_id = do_add_node(new ParameterNode(p));
    parameter_nodes.push_back(new_node_id); // need to add to extra parameter_nodes
    return new_node_id;
}
inline
node_id_t ComputationGraph::add_parameter(const LookupParameter& lp)
{
    node_id_t new_node_id = do_add_node(new ParameterNode(lp));
    parameter_nodes.push_back(new_node_id); 
    return new_node_id;
}

inline
node_id_t ComputationGraph::add_const_parameter(const Parameter& cp)
{
    return do_add_node(new ConstParameterNode(cp));
}
inline
node_id_t ComputationGraph::add_const_parameter(const LookupParameter& clp)
{
    return do_add_node(new ConstParameterNode(clp));
}


inline
node_id_t ComputationGraph::add_lookup(const LookupParameter& l, unsigned i)
{
    auto new_node_id = do_add_node(new LookupNode(l, i));
    parameter_nodes.push_back(new_node_id);
    return new_node_id;
}

inline
node_id_t ComputationGraph::add_lookup(const LookupParameter& l, const unsigned* pi)
{
    auto new_node_id = do_add_node(new LookupNode(l, pi));
    parameter_nodes.push_back(new_node_id);
    return new_node_id;
}

inline
node_id_t ComputationGraph::add_lookup(const LookupParameter& l, const std::vector<unsigned>& indices)
{
    auto new_node_id = do_add_node(new LookupNode(l, indices));
    parameter_nodes.push_back(new_node_id);
    return new_node_id;
}

inline
node_id_t ComputationGraph::add_lookup(const LookupParameter& l, const std::vector<unsigned>* pindices)
{
    auto new_node_id = do_add_node(new LookupNode(l, pindices));
    parameter_nodes.push_back(new_node_id);
    return new_node_id;
}


inline
node_id_t ComputationGraph::add_const_lookup(const LookupParameter& l, unsigned i)
{
    return do_add_node(new LookupNode(l, i));
}
inline
node_id_t ComputationGraph::add_const_lookup(const LookupParameter& l, const unsigned* pi)
{
    return do_add_node(new LookupNode(l, pi));
}
inline
node_id_t ComputationGraph::add_const_lookup(const LookupParameter& l, const std::vector<unsigned>& indices)
{
    return do_add_node(new LookupNode(l, indices));
}
inline
node_id_t ComputationGraph::add_const_lookup(const LookupParameter& l, const std::vector<unsigned>* pindices)
{
    return do_add_node(new LookupNode(l, pindices));
}


template<typename Function>
inline
node_id_t ComputationGraph::add_function(const std::initializer_list<node_id_t>& args)
{
    return do_add_node(new Function(args));
}

template<typename Function, typename T>
inline
node_id_t ComputationGraph::add_function(const T& args)
{
    return do_add_node(new Function(args));
}

template<typename Function, typename... Args>
inline
node_id_t ComputationGraph::add_function(const std::initializer_list<node_id_t>& args,
                                         Args&&... side_informations)
{
    return do_add_node(new Function(args, std::forward<Args>(side_information)...));
}


template<typename Function, typename T, typename... Args>
inline
node_id_t ComputationGraph::add_function(const T&args, Args&&... side_infomation)
{
    return do_add_node(new Function(args, std::forward<Args>(side_information)...));
}

inline
const Dim& ComputationGraph::get_dimention(node_id_t i) const
{
    return nodes[i]->get_dimention();
}

inline
void Node::forward(const std::vector<const Tensor*>& xs,
                   Tensor& fx) const
{
    /**
        no support for batch, so just call forward_impl
    */
    return forward_impl(xs, fx);
}

inline
void Node::backward(const std::vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const
{
    return backward_impl(xs, fx, dEdf, i, dEdxi);
}


} // end of namespace symnn
#endif