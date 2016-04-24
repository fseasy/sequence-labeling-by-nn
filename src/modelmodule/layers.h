#ifndef LAYERS_H_INCLUDE
#define LAYERS_H_INCLUDE

#include <vector>

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using std::vector ;
using cnn::LSTMBuilder; using cnn::Parameters; using cnn::Model; using cnn::expr::Expression;
using cnn::ComputationGraph;

namespace slnn {

struct BILSTMLayer
{
    LSTMBuilder *l2r_builder;
    LSTMBuilder *r2l_builder;
    Parameters *SOS;
    Parameters *EOS;
    Expression SOS_EXP;
    Expression EOS_EXP;

    BILSTMLayer(Model *model , unsigned nr_lstm_stack_layers, unsigned lstm_x_dim, unsigned lstm_h_dim);
    ~BILSTMLayer();
    void new_graph(ComputationGraph &cg);
    void start_new_sequence();
    void build_graph(const vector<Expression> &X_seq , vector<Expression> &l2r_outputs , 
        vector<Expression> &r2l_outputs);

};

struct DenseLayer
{
    Parameters *w,
        *b;
    Expression w_exp,
        b_exp;
    DenseLayer(Model *m , unsigned input_dim , unsigned output_dim );
    ~DenseLayer();
    inline void new_graph(ComputationGraph &cg);
    inline Expression build_graph(Expression &e);
};

struct Merge2Layer
{
    Parameters *w1,
        *w2,
        *b;
    Expression w1_exp,
        w2_exp,
        b_exp;
    Merge2Layer(Model *model , unsigned input1_dim, unsigned input2_dim, unsigned output_dim );
    ~Merge2Layer();
    inline void new_graph(ComputationGraph &cg);
    inline Expression build_graph(Expression &e1, Expression &e2);
};

struct Merge3Layer
{
    Parameters *w1 ,
        *w2 , 
        *w3 ,
        *b;
    Expression w1_exp,
        w2_exp,
        w3_exp,
        b_exp;
    Merge3Layer(Model *model ,unsigned input1_dim , unsigned input2_dim , unsigned input3_dim , unsigned output_dim);
    ~Merge3Layer();
    inline void new_graph(ComputationGraph &cg);
    inline Expression build_graph(Expression &e1, Expression &e2, Expression &e3);
};




// ------------------- inline function definition --------------------

inline
void BILSTMLayer::new_graph(ComputationGraph &cg)
{
    l2r_builder->new_graph(cg);
    r2l_builder->new_graph(cg);
    SOS_EXP = parameter(cg, SOS);
    EOS_EXP = parameter(cg, EOS);
}

inline
void BILSTMLayer::start_new_sequence()
{
    l2r_builder->start_new_sequence();
    r2l_builder->start_new_sequence();
}

inline
void BILSTMLayer::build_graph(const vector<Expression> &X_seq, vector<Expression> &l2r_outputs,
    vector<Expression> &r2l_outputs)
{
    size_t seq_len = X_seq.size();
    vector<Expression> tmp_l2r_outputs(seq_len),
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

// DenseLayer
inline 
void DenseLayer::new_graph(ComputationGraph &cg)
{
    w_exp = parameter(cg, w);
    b_exp = parameter(cg, b);
}
inline
Expression DenseLayer::build_graph(Expression &e)
{
    return affine_transform({ 
       b_exp ,
       w_exp , e 
    });
}

// Merge2Layer 
inline 
void Merge2Layer::new_graph(ComputationGraph &cg)
{
    b_exp = parameter(cg, b);
    w1_exp = parameter(cg, w1);
    w2_exp = parameter(cg, w2);
}
inline
Expression Merge2Layer::build_graph(Expression &e1, Expression &e2)
{
    return affine_transform({
        b_exp ,
        w1_exp , e1,
        w2_exp , e2,
    });
}

// Merge3Layer
inline 
void Merge3Layer::new_graph(ComputationGraph &cg)
{
    b_exp = parameter(cg, b);
    w1_exp = parameter(cg, w1);
    w2_exp = parameter(cg, w2);
    w3_exp = parameter(cg, w3);

}
inline
Expression Merge3Layer::build_graph(Expression &e1, Expression &e2, Expression &e3)
{
    return affine_transform({
        b_exp,
        w1_exp, e1 ,
        w2_exp, e2 ,
        w3_exp, e3
    });
}

}


#endif
