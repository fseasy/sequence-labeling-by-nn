#ifndef SYMNN_EXPR_H_INCLUDED_
#define SYMNN_EXPR_H_INCLUDED_

#include "symnn/symnn.h"

namespace symnn {
namespace expr {

class Expression
{
    friend class ComputationGraph;
public:
    ComputationGraph* get_cg() { return pcg; }
    const ComputationGraph* get_cg() const { return pcg; }
private:
    ComputationGraph *pcg;
    node_id_t node_id;
};





} // end of namespace expr
} // end of namespace symnn


#endif