#include "symnn/symnn.h"
#include "symnn/exec.h"
#include "symnn/param_nodes.h"
#include "symnn/expr.h"

using namespace std;

namespace symnn {



ComputationGraph::~ComputationGraph()
{
    clear();
    delete engine;
}

void ComputationGraph::clear()
{
    parameter_nodes.clear();
    for (auto node : nodes) { delete node; }
    nodes.clear();
    engine->invalidate();
}

void ComputationGraph::set_dim_cg4new_node(node_id_t new_node_id)
{
    Node* pnode = nodes.at(new_node_id);
    std::vector<const Dim*> arg_dims(pnode->arity()));
    std::size_t pos = 0;
    for (node_id_t arg_node_id : pnode->args)
    {
        Node* parg_node = nodes.at(arg_node_id);
        arg_dims[pos++] = &parg_node->get_dimention();
    }
    pnode->dim = pnode->dim_forward(arg_dims);
    pnode->pcg = this;
}

/***
 * because ComputationGraph and Expression are co-dependency,
 * in order to use inline in Expression,
 * here we can't use inline.
 */
const Tensor& ComputationGraph::forward(const expr::Expression& last)
{
    return forward(last.node_id);
}
void ComputationGraph::backward(const expr::Expression& last)
{
    return backward(last.node_id);
}

const Tensor& ComputationGraph::get_value(const expr::Expression& e)
{
    return get_value(e.node_id);
}

const Tensor& ComputationGraph::get_gradient(const expr::Expression& e)
{
    return get_gradient(e.node_id);
}



} // end of namespace symnn