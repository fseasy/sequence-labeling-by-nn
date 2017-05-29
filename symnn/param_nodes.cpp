#include <string>
#include <sstream>
#include <vector>

#include "symnn/param_nodes.h"
#include "misc/except_macros.h"

using namespace std;

namespace symnn {

string ConstParameterNode::as_string() const
{
    ostringstream oss;
    oss << "ConstParameterNode {" << dim << ", @" << params.get() << "}";
    return oss.str();
}

Dim ConstParameterNode::dim_forward(const vector<const Dim*>& arg_dims) const
{
    SLNN_ASSERT(arg_dims.size() == 0U, MODULE_SYMNN_NAME,
                "Failed to check args dimension in ConstParameterNode");
    return dim;
}

string ParameterNode::as_string() const
{
    ostringstream oss;
    oss << "ParameterNode {" << dim << ", @" << params.get() << "}";
    return oss.str();
}
Dim ParameterNode::dim_forward(const vector<const Dim*>& arg_dims) const
{
    SLNN_ASSERT(arg_dims.size() == 0U, MODULE_SYMNN_NAME,
                "Failed to check args dimension in ParameterNode");
    return dim;
}
void ParameterNode::accumulate_grad(const Tensor& g)
{
    if (params.get_model() != nullptr)
    {
        return params.get()->accumulate_grad(g);
    }
    else if (lparams.get_model() != nullptr)
    {
        return lparams.get()->accumulate_grad(g);
    }
    else 
    {
        SLNN_RUNTIME_ERROR(MODULE_SYMNN_NAME,
                           "none of param or lookupparam has been used "
                           "in ParameterNode.");
    }
}


string InputNode::as_string() const
{
    ostringstream oss;
    oss << "InputNode {" << dim << ", @" << pdata << "}";
    return oss.str();
}
Dim InputNode::dim_forward(const vector<const Dim*>& arg_dims) const
{
    SLNN_ASSERT(arg_dims.size() == 0U, MODULE_SYMNN_NAME,
                "Failed to check args dimension in InputNode");
    return dim;
}

string ScalarInputNode::as_string() const
{
    ostringstream oss;
    oss << "ScalarInputNode {" << Dim({1}) << ", @" << pdata << "}";
    return oss.str();
}
Dim ScalarInputNode::dim_forward(const vector<const Dim*>& arg_dims) const
{
    SLNN_ASSERT(arg_dims.size() == 0U, MODULE_SYMNN_NAME,
                "Failed to check args dimension in ScalarInputNode");
    return Dim({1});
}

string LookupNode::as_string() const
{
    ostringstream oss;
    oss << "LookupNode{" << params.get()->get_all_dimension()
        << "-->" << dim << "}";
    return oss.str();
}
Dim LookupNode::dim_forward(const vector<const Dim*>& arg_dims) const
{
    SLNN_ASSERT(arg_dims.size() == 0U, MODULE_SYMNN_NAME,
                "Failed to check args dimension in LookNode");
    return dim;
}
void LookupNode::accumulate_grad(const Tensor& g)
{
    if (pindex)
    {
        return params.get()->accumulate_grad(*pindex, g);
    }
    /* TODO */
    else 
    {
        SLNN_RUNTIME_ERROR(MODULE_SYMNN_NAME,
                           "un-finished implementation.");
    }
}


} // end of namespace symnn