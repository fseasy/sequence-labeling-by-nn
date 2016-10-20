#ifndef LAYERS_H_INCLUDE
#define LAYERS_H_INCLUDE

#include <vector>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/gru.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "utils/typedeclaration.h"

namespace slnn {

template<typename RNNDerived>
struct BIRNNLayer
{
    RNNDerived *l2r_builder;
    RNNDerived *r2l_builder;
    dynet::Parameter SOS;
    dynet::Parameter EOS;
    dynet::expr::Expression SOS_EXP;
    dynet::expr::Expression EOS_EXP;
    dynet::real default_dropout_rate ;

    BIRNNLayer(dynet::Model *model , unsigned nr_rnn_stack_layers, unsigned rnn_x_dim, unsigned  rnn_h_dim ,
                dynet::real default_dropout_rate=0.f);
    ~BIRNNLayer();
    void new_graph(dynet::ComputationGraph &cg);
    void set_dropout(float dropout_rate) ;
    void set_dropout();
    void disable_dropout() ;
    void start_new_sequence();
    void build_graph(const std::vector<dynet::expr::Expression> &X_seq , std::vector<dynet::expr::Expression> &l2r_outputs , 
                     std::vector<dynet::expr::Expression> &r2l_outputs);
};

using BILSTMLayer = BIRNNLayer<dynet::LSTMBuilder>;
using BISimpleRNNLayer = BIRNNLayer<dynet::SimpleRNNBuilder>;
using BIGRULayer = BIRNNLayer<dynet::GRUBuilder>;

struct DenseLayer
{
    dynet::Parameter w,
        b;
    dynet::expr::Expression w_exp,
        b_exp;
    DenseLayer(dynet::Model *m , unsigned input_dim , unsigned output_dim );
    ~DenseLayer();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_graph(const dynet::expr::Expression &e);
};

struct Merge2Layer
{
    dynet::Parameter w1,
        w2,
        b;
    dynet::expr::Expression w1_exp,
        w2_exp,
        b_exp;
    Merge2Layer(dynet::Model *model , unsigned input1_dim, unsigned input2_dim, unsigned output_dim );
    ~Merge2Layer();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2);
};

struct Merge3Layer
{
    dynet::Parameter w1 ,
        w2 , 
        w3 ,
        b;
    dynet::expr::Expression w1_exp,
        w2_exp,
        w3_exp,
        b_exp;
    Merge3Layer(dynet::Model *model ,unsigned input1_dim , unsigned input2_dim , unsigned input3_dim , unsigned output_dim);
    ~Merge3Layer();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2, const dynet::expr::Expression &e3);
};

struct Merge4Layer
{
    dynet::Parameter w1 ,
        w2, 
        w3,
        w4,
        b;
    dynet::expr::Expression w1_exp,
        w2_exp,
        w3_exp,
        w4_exp,
        b_exp;
    Merge4Layer(dynet::Model *model ,unsigned input1_dim , unsigned input2_dim , unsigned input3_dim , unsigned input4_dim, unsigned output_dim);
    ~Merge4Layer();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2, const dynet::expr::Expression &e3,
        const dynet::expr::Expression &e4);
};

struct MLPHiddenLayer
{
    // data
    unsigned nr_hidden_layer;
    std::vector<dynet::Parameter> w_list;
    std::vector<dynet::Parameter> b_list;
    std::vector<dynet::expr::Expression> w_expr_list;
    std::vector<dynet::expr::Expression> b_expr_list;
    dynet::real dropout_rate;
    bool is_enable_dropout;
    NonLinearFunc *nonlinear_func;
    // constructor
    MLPHiddenLayer(dynet::Model *m, unsigned input_dim, const std::vector<unsigned> &hidden_layer_dim_list, 
        dynet::real dropout_rate=0.f,
        NonLinearFunc *nonlinear_func=dynet::expr::tanh);
    // interface
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression
        build_graph(const dynet::expr::Expression &input_expr);
    void build_graph(const std::vector<dynet::expr::Expression> &input_exprs, std::vector<dynet::expr::Expression> &output_exprs);
    // setter
    void enable_dropout(){ is_enable_dropout = true; }
    void disable_dropout(){ is_enable_dropout = false; }
};


// ------------------- inline function definition --------------------
// DenseLayer
inline 
void DenseLayer::new_graph(dynet::ComputationGraph &cg)
{
    w_exp = parameter(cg, w);
    b_exp = parameter(cg, b);
}
inline
Expression DenseLayer::build_graph(const dynet::expr::Expression &e)
{
    return affine_transform({ 
       b_exp ,
       w_exp , e 
    });
}

// Merge2Layer 
inline 
void Merge2Layer::new_graph(dynet::ComputationGraph &cg)
{
    b_exp = parameter(cg, b);
    w1_exp = parameter(cg, w1);
    w2_exp = parameter(cg, w2);
}
inline
dynet::expr::Expression Merge2Layer::build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2)
{
    return affine_transform({
        b_exp ,
        w1_exp , e1,
        w2_exp , e2,
    });
}

// Merge3Layer
inline 
void Merge3Layer::new_graph(dynet::ComputationGraph &cg)
{
    b_exp = parameter(cg, b);
    w1_exp = parameter(cg, w1);
    w2_exp = parameter(cg, w2);
    w3_exp = parameter(cg, w3);

}
inline
dynet::expr::Expression Merge3Layer::build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2, const dynet::expr::Expression &e3)
{
    return affine_transform({
        b_exp,
        w1_exp, e1 ,
        w2_exp, e2 ,
        w3_exp, e3
    });
}

// Merge4Layer
inline 
void Merge4Layer::new_graph(dynet::ComputationGraph &cg)
{
    b_exp = parameter(cg, b);
    w1_exp = parameter(cg, w1);
    w2_exp = parameter(cg, w2);
    w3_exp = parameter(cg, w3);
    w4_exp = parameter(cg, w4);
}

inline 
dynet::expr::Expression Merge4Layer::build_graph(const dynet::expr::Expression &e1, const dynet::expr::Expression &e2,
    const dynet::expr::Expression &e3, const dynet::expr::Expression &e4)
{
    return affine_transform({
        b_exp,
        w1_exp, e1 ,
        w2_exp, e2 ,
        w3_exp, e3,
        w4_exp, e4
    });
}

// MLPHiddenLayer

inline
void MLPHiddenLayer::new_graph(dynet::ComputationGraph &cg)
{
    for( unsigned i = 0 ; i < nr_hidden_layer; ++i )
    {
        w_expr_list[i] = parameter(cg, w_list[i]);
        b_expr_list[i] = parameter(cg, b_list[i]);
    }
}

inline
dynet::expr::Expression
MLPHiddenLayer::build_graph(const dynet::expr::Expression &input_expr)
{
    dynet::expr::Expression tmp_expr = input_expr;
    for( unsigned i = 0 ; i < nr_hidden_layer; ++i )
    {
        dynet::expr::Expression net_expr = affine_transform({
            b_expr_list[i],
            w_expr_list[i], tmp_expr });
        if( is_enable_dropout && std::abs(dropout_rate - 0.f) > 1e-6 ) { net_expr = dynet::expr::dropout(net_expr, dropout_rate); }
        tmp_expr = (*nonlinear_func)(net_expr);
    }
    return tmp_expr;
}

inline
void MLPHiddenLayer::build_graph(const std::vector<dynet::expr::Expression> &input_exprs,
    std::vector<dynet::expr::Expression> &output_exprs)
{
    unsigned sz = input_exprs.size();
    std::vector<dynet::expr::Expression> tmp_output_exprs(sz);
    for( unsigned i = 0; i < sz; ++i )
    {
        tmp_output_exprs.at(i) = build_graph(input_exprs.at(i));
    }
    swap(output_exprs, tmp_output_exprs);
}

/*****************************
*    Template Implementation
*****************************/

template<typename RNNDerived> 
BIRNNLayer<RNNDerived>::BIRNNLayer(dynet::Model *m , unsigned nr_rnn_stacked_layers, unsigned rnn_x_dim, unsigned rnn_h_dim ,
                                   dynet::real default_dropout_rate)
    : l2r_builder(new RNNDerived(nr_rnn_stacked_layers , rnn_x_dim , rnn_h_dim , m)) ,
    r2l_builder(new RNNDerived(nr_rnn_stacked_layers , rnn_x_dim , rnn_h_dim , m)) ,
    SOS(m->add_parameters({rnn_x_dim})) ,
    EOS(m->add_parameters({rnn_x_dim})) ,
    default_dropout_rate(default_dropout_rate)
{}

template<typename RNNDerived>
BIRNNLayer<RNNDerived>::~BIRNNLayer()
{ 
    delete l2r_builder;
    delete r2l_builder;
}

template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::new_graph(dynet::ComputationGraph &cg)
{
    l2r_builder->new_graph(cg);
    r2l_builder->new_graph(cg);
    SOS_EXP = parameter(cg, SOS);
    EOS_EXP = parameter(cg, EOS);
}

template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::set_dropout(float dropout_rate)
{
    l2r_builder->set_dropout(dropout_rate) ;
    r2l_builder->set_dropout(dropout_rate) ;
}

// SimpleRNNBuilder , GRUBuilder has no function `set_dropout(float)`
template <>
inline
void BIRNNLayer<dynet::SimpleRNNBuilder>::set_dropout(float){ } // empty implementation
template <>
inline
void BIRNNLayer<dynet::GRUBuilder>::set_dropout(float){}


template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::set_dropout()
{
    l2r_builder->set_dropout(default_dropout_rate) ;
    r2l_builder->set_dropout(default_dropout_rate) ;
}
// SimpleRNNBuilder, GRUBuilder has no function `set_dropout(float)`
template <>
inline 
void BIRNNLayer<dynet::SimpleRNNBuilder>::set_dropout(){}
template <>
inline
void BIRNNLayer<dynet::GRUBuilder>::set_dropout(){}

template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::disable_dropout()
{
    l2r_builder->disable_dropout() ;
    r2l_builder->disable_dropout() ;
}
// SimpleRNNBuilder , GRUBulider has no function `disable_dropout()`
template <>
inline 
void BIRNNLayer<dynet::SimpleRNNBuilder>::disable_dropout(){}
template <>
inline
void BIRNNLayer<dynet::GRUBuilder>::disable_dropout(){}


template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::start_new_sequence()
{
    l2r_builder->start_new_sequence();
    r2l_builder->start_new_sequence();
}

template <typename RNNDerived>
inline
void BIRNNLayer<RNNDerived>::build_graph(const std::vector<dynet::expr::Expression> &X_seq, std::vector<dynet::expr::Expression> &l2r_outputs,
                                         std::vector<dynet::expr::Expression> &r2l_outputs)
{
    size_t seq_len = X_seq.size();
    std::vector<dynet::expr::Expression> tmp_l2r_outputs(seq_len),
        tmp_r2l_outputs(seq_len);
    l2r_builder->add_input(SOS_EXP);
    r2l_builder->add_input(EOS_EXP);
    for (int pos = 0; pos < static_cast<int>(seq_len); ++pos)
    {
        tmp_l2r_outputs[pos] = l2r_builder->add_input(X_seq[pos]);
        int reverse_pos = seq_len - pos - 1;
        tmp_r2l_outputs[reverse_pos] = r2l_builder->add_input(X_seq[reverse_pos]);
    }
    swap(l2r_outputs, tmp_l2r_outputs);
    swap(r2l_outputs, tmp_r2l_outputs);
}



} // end of namespace slnn


#endif
