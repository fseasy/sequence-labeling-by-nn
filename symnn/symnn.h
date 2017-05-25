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

namespace symnn{

// pre-declaring.
// for decoupling the dependency.
class Node;
class ExecutionEngine;
namespace expr { class Expression; }

class ComputationGraph
{
public:
    // constructor / deconstructor
    ComputationGraph();
    ~ComputationGraph();

    // inputs
    node_id_t add_input(real_t i);

    // parameters
    node_id_t add_parameter(Parameter p);
    node_id_t add_const_parameter(const Parameter p);

    // functions
    template<typename Function>
    inline node_id_t add_function(
        const std::initializer_list<node_id_t>& arguments);
    
    // flow
    const Tensor& forward(const expr::Expression& last);
    void backward(const expr::Expression& last);

private:
    std::vector<Node*> nodes; // in topological order
    std::vector<node_id_t> parameter_nodes; // the parameters nodes (can be update)

};

} // end of namespace symnn
#endif