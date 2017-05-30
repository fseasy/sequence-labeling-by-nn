#ifndef SYMNN_EXEC_H_INCLUDED_
#define SYMNN_EXEC_H_INCLUDED_

#include "symnn/type.h"

class ComputationGraph;

namespace symnn {

class ExecutionEngine
{
public:
    virtual ~ExecutionEngine();
    virtual const Tensor& forward(node_id_t i) = 0;
    virtual const Tensor& incremental_forward() = 0;
    virtual const Tensor& incremental_forward(node_id_t i) = 0;
    virtual void backward(node_id_t i) = 0;
    virtual void invalidate() = 0;
    virtual void invalidate(unsigned) = 0;
protected:
    explicit ExecutionEngine(const ComputationGraph* pcg) :pcg(pcg) {}

protected:
    const ComputationGraph* pcg;
};

class SimpleExcutionEngine: public ExecutionEngine
{
public:
    explicit SimpleExcutionEngine(const ComputationGraph* pcg) : ExecutionEngine(pcg) {}
    

    const Tensor& forward(node_id_t i) override;
    const Tensor& incremental_forward() override;
    const Tensor& incremental_forward(node_id_t i) override;
    void backward(node_id_t i) override;

    virtual void invalidate() override;
    virtual void invalidate(unsigned) override;
private:
    std::vector<Tensor> nfxs;
    std::vector<Tensor> ndEdfs;
    node_id_t num_nodes_evaluated;
};


} // end of namespace symnn


#endif