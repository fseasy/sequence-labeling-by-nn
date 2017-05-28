#ifndef SYMNN_NODES_MACROS_H_INCLUDED_
#define SYMNN_NODES_MACROS_H_INCLUDED_

//namespace symnn{

#define NODE_DEFINE_FLOW_IMPL() \
    const Dim& dim_forward(const std::vector<const Dim*>& xs) const override; \
    void forward_impl(const std::vector<const Tensor*>& xs,            \
                      Tensor& fx) const override;                      \
    void backward_impl(const std::vector<const Tensor*>& xs,           \
                       const Tensor& fx,                               \
                       const Tensor& dEdf,                             \
                       unsigned i,                                     \
                       Tensor& dEdxi) const overide;                   



//} // end of namespace symnn


#endif