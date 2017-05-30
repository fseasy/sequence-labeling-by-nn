#ifndef SYMNN_EXPR_H_INCLUDED_
#define SYMNN_EXPR_H_INCLUDED_

#include "symnn/symnn.h"

namespace symnn {
namespace expr {

class Expression
{
    friend class ComputationGraph;
public:

    Expression();
    Expression(ComputationGraph *pcg, node_id_t i);

    ComputationGraph* get_cg() { return pcg; }
    const ComputationGraph* get_cg() const { return pcg; }
private:
    ComputationGraph *pcg;
    node_id_t node_id;
};


/**
 * inline implementaion
 *******/
inline
Expression::Expression()
    :pcg(nullptr), node_id(0) {}

inline
Expression::Expression(ComputationGraph *pcg, node_id_t i)
    : pcg(pcg), node_id(i) {}



} // end of namespace expr
} // end of namespace symnn


#endif