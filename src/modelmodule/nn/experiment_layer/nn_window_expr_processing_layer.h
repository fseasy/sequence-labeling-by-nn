#ifndef SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_EXP_LAYER_WINDOW_EXPR_PRO_LAYER_H_
#define SLNN_SEGMENTER_CWS_MODULE_NN_MODULE_EXP_LAYER_WINDOW_EXPR_PRO_LAYER_H_
#include <vector>
#include <deque>
#include <memory>
#include <string>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
namespace slnn{
namespace module{
namespace nn{
namespace experiment{


class WindowExprProcessingLayerAbstract
{
public:
    virtual void new_graph(dynet::ComputationGraph &rcg){};
    virtual unsigned get_output_dim() const = 0;
    virtual std::vector<dynet::expr::Expression> 
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) = 0;
    virtual ~WindowExprProcessingLayerAbstract(){};
};

class WindowExprConcatLayer : public WindowExprProcessingLayerAbstract
{
public:
    WindowExprConcatLayer(unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprConcatLayer(const WindowExprConcatLayer&) = delete;
    WindowExprConcatLayer& operator=(const WindowExprConcatLayer&) = delete;
public:
    unsigned get_output_dim() const override { return output_dim; };
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    unsigned output_dim;
};

class WindowExprSumLayer : public WindowExprProcessingLayerAbstract
{
public:
    WindowExprSumLayer(unsigned unit_embedding_dim);
    WindowExprSumLayer(const WindowExprSumLayer&) = delete;
    WindowExprSumLayer& operator=(const WindowExprSumLayer&) = delete;
public:
    unsigned get_output_dim() const override{ return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    unsigned output_dim;
};

class WindowExprBigramLayer : public WindowExprProcessingLayerAbstract
{
    // reference: YU L, HERMANN K M, BLUNSOM P, PULMAN S£¬. 
    // Deep Learning for Answer Sentence Selection[J]. NIPS deep learning workshop, 2014: 9.
public:
    WindowExprBigramLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprBigramLayer(const WindowExprBigramLayer&) = delete;
    WindowExprBigramLayer& operator=(const WindowExprBigramLayer&) = delete;
public:
    virtual void new_graph(dynet::ComputationGraph &rcg) override;
    unsigned get_output_dim() const override{ return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    dynet::Parameter Tl_param;
    dynet::Parameter Tr_param;
    dynet::Parameter b_param;
    dynet::expr::Expression Tl_expr;
    dynet::expr::Expression Tr_expr;
    dynet::expr::Expression b_expr;
    std::deque<dynet::expr::Expression> cached_c_expr_list;
    unsigned output_dim;
};


class WindowExprBigramConcatLayer : public WindowExprProcessingLayerAbstract
{
    // variant to bigram model
public:
    WindowExprBigramConcatLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprBigramConcatLayer(const WindowExprBigramConcatLayer&) = delete;
    WindowExprBigramConcatLayer& operator=(const WindowExprBigramConcatLayer&) = delete;
public:
    void new_graph(dynet::ComputationGraph &rcg) override;
    unsigned get_output_dim() const override{ return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    dynet::Parameter Tl_param;
    dynet::Parameter Tr_param;
    dynet::Parameter b_param;
    dynet::expr::Expression Tl_expr;
    dynet::expr::Expression Tr_expr;
    dynet::expr::Expression b_expr;
    std::deque<dynet::expr::Expression> cached_c_expr_list;
    unsigned output_dim;
};

class WindowExprAttention1Layer : public WindowExprProcessingLayerAbstract
{
public:
    WindowExprAttention1Layer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprAttention1Layer(const WindowExprAttention1Layer&) = delete;
    WindowExprAttention1Layer& operator=(const WindowExprAttention1Layer&) = delete;
public:
    void new_graph(dynet::ComputationGraph &rcg) override;
    unsigned get_output_dim() const override{ return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    dynet::Parameter W;
    dynet::Parameter U;
    dynet::Parameter v;
    dynet::expr::Expression W_expr;
    dynet::expr::Expression U_expr;
    dynet::expr::Expression v_expr;
    dynet::ComputationGraph *pcg;
    unsigned output_dim;
    unsigned window_sz;
};


class WindowExprAttention1SumLayer : public WindowExprProcessingLayerAbstract
{
public:
    WindowExprAttention1SumLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprAttention1SumLayer(const WindowExprAttention1SumLayer&) = delete;
    WindowExprAttention1SumLayer& operator=(const WindowExprAttention1SumLayer&) = delete;
public:
    void new_graph(dynet::ComputationGraph &rcg) override;
    unsigned get_output_dim() const override{ return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list) override;
private:
    dynet::Parameter W;
    dynet::Parameter U;
    dynet::Parameter v;
    dynet::expr::Expression W_expr;
    dynet::expr::Expression U_expr;
    dynet::expr::Expression v_expr;
    dynet::ComputationGraph *pcg;
    unsigned output_dim;
    unsigned window_sz;
};

class WindowExprBiLstmLayer : public WindowExprProcessingLayerAbstract
{
public:
    WindowExprBiLstmLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);
    WindowExprBiLstmLayer(const WindowExprBiLstmLayer&) = delete;
    WindowExprBiLstmLayer& operator=(const WindowExprBiLstmLayer&) = delete;
public:
    void new_graph(dynet::ComputationGraph &rcg) override;
    unsigned get_output_dim() const override { return output_dim; }
    std::vector<dynet::expr::Expression>
        process(const std::vector<std::vector<dynet::expr::Expression>> &window_expr_list) override;
private:
    dynet::LSTMBuilder l2r_builder;
    dynet::LSTMBuilder r2l_builder;
    unsigned output_dim;
    unsigned window_sz;
};

std::shared_ptr<WindowExprProcessingLayerAbstract>
create_window_expr_processing_layer(const std::string& processing_method,
    dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz);


/*****************************************************
* Inline Implementation
******************************************************/


/***********
 * Window Expression Concatenate layer
 ***********/
inline
std::vector<dynet::expr::Expression>
WindowExprConcatLayer::process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list)
{
    unsigned len = window_expr_list.size();
    std::vector<dynet::expr::Expression> concat_expr_list(len);
    for( unsigned i = 0; i < len; ++i )
    {
        concat_expr_list[i] = dynet::expr::concatenate(window_expr_list[i]);
    }
    return concat_expr_list;
}

/*************
 * Window Expression Sum(average) Layer
 *************/

inline
std::vector<dynet::expr::Expression>
WindowExprSumLayer::process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list)
{
    unsigned len = window_expr_list.size();
    std::vector<dynet::expr::Expression> sum_expr_list(len);
    for( unsigned i = 0; i < len; ++i )
    {
        sum_expr_list[i] = dynet::expr::sum(window_expr_list[i]);
    }
    return sum_expr_list;
}


/*************
 * Window Expression Bigram Layer
 *************/

inline
void WindowExprBigramLayer::new_graph(dynet::ComputationGraph &rcg)
{
    Tl_expr = dynet::expr::parameter(rcg, Tl_param);
    Tr_expr = dynet::expr::parameter(rcg, Tr_param);
    b_expr = dynet::expr::parameter(rcg, b_param);
}

/**************
 * Window Expression Bigram by Concat Layer
 **************/

inline
void WindowExprBigramConcatLayer::new_graph(dynet::ComputationGraph &rcg)
{
    Tl_expr = dynet::expr::parameter(rcg, Tl_param);
    Tr_expr = dynet::expr::parameter(rcg, Tr_param);
    b_expr = dynet::expr::parameter(rcg, b_param);
}

/************
 * Window Expression Attention 1 Layer
 ************/
inline
void WindowExprAttention1Layer::new_graph(dynet::ComputationGraph &rcg)
{
    pcg = &rcg;
    W_expr = dynet::expr::parameter(rcg, W);
    U_expr = dynet::expr::parameter(rcg, U);
    v_expr = dynet::expr::parameter(rcg, v);
}

/**************
 * Window Expression Attention 1 (sum) Layer
 **************/
inline
void WindowExprAttention1SumLayer::new_graph(dynet::ComputationGraph &rcg)
{
    pcg = &rcg;
    W_expr = dynet::expr::parameter(rcg, W);
    U_expr = dynet::expr::parameter(rcg, U);
    v_expr = dynet::expr::parameter(rcg, v);
}


} // end of namespace experiment
} // end of namespace nn
} // end of namespace module
} // end of namespace slnn



#endif