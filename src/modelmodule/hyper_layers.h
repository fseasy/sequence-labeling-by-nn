#ifndef HYPER_LAYERS_H_INCLUDED_
#define HYPER_LAYERS_H_INCLUDED_

#include "layers.h"
#include "utils/typedeclaration.h"

namespace slnn{

struct Input1
{
    cnn::LookupParameters *word_lookup_param;
    cnn::ComputationGraph *pcg;
    Input1(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim);
    ~Input1();
    void new_graph(cnn::ComputationGraph &cg);
    void build_inputs(const IndexSeq &sent , std::vector<cnn::expr::Expression> &input_exprs);
};

struct Input2D
{
    cnn::LookupParameters *dynamic_lookup_param1,
        *dynamic_lookup_param2;
    Merge2Layer m2_layer;
    cnn::ComputationGraph *pcg;
    NonLinearFunc *nonlinear_func;
    Input2D(cnn::Model *m, unsigned vocab_size1, unsigned embedding_dim1,
        unsigned vocab_size2, unsigned embedding_dim2,
        unsigned mergeout_dim ,
        NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    ~Input2D();
    void new_graph(cnn::ComputationGraph &cg);
    void build_inputs(const IndexSeq &sent1, const IndexSeq &sent2, std::vector<cnn::expr::Expression> &inputs_exprs);
};

struct Input2
{
    cnn::LookupParameters *dynamic_lookup_param,
        *fixed_lookup_param;
    Merge2Layer m2_layer;
    cnn::ComputationGraph *pcg;
    NonLinearFunc *nonlinear_func;

    Input2(cnn::Model *m, unsigned dynamic_vocab_size, unsigned dynamic_embedding_dim,
        unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
        unsigned mergeout_dim , NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    ~Input2();
    void new_graph(cnn::ComputationGraph &cg);
    void build_inputs(const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent, 
        std::vector<cnn::expr::Expression> &inputs_exprs);
};

struct Input3
{
    cnn::LookupParameters *dynamic_lookup_param1,
        *dynamic_lookup_param2 ,
        *fixed_lookup_param;
    Merge3Layer m3_layer;
    cnn::ComputationGraph *pcg;
    NonLinearFunc *nonlinear_func;
    Input3(cnn::Model *m, unsigned dynamic_vocab_size1, unsigned dynamic_embedding_dim1 ,
        unsigned dynamic_vocab_size2 , unsigned dynamic_embedding_dim2 , 
        unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
        unsigned mergeout_dim ,
        NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    ~Input3();
    void new_graph(cnn::ComputationGraph &cg);
    void build_inputs(const IndexSeq &dynamic_sent1, const IndexSeq &dynamic_sent2 , const IndexSeq &fixed_sent,
        std::vector<cnn::expr::Expression> &inputs_exprs);
};


struct SimpleOutput
{
    Merge2Layer hidden_layer;
    DenseLayer output_layer;
    NonLinearFunc *nonlinear_func;
    cnn::ComputationGraph *pcg;
    SimpleOutput(cnn::Model *m, unsigned input_dim1, unsigned input_dim2 ,
        unsigned hidden_dim, unsigned output_dim , NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    ~SimpleOutput();
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
           const std::vector<cnn::expr::Expression> &expr_cont2 ,
           const IndexSeq &gold_seq);
    void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        IndexSeq &pred_out_seq);
};

struct PretagOutput
{
    Merge3Layer hidden_layer;
    DenseLayer output_layer;
    NonLinearFunc *nonlinear_func;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::Parameters *TAG_SOS ;
    cnn::ComputationGraph *pcg;


    PretagOutput(cnn::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
                 unsigned hidden_dim, unsigned output_dim , 
                 NonLinearFunc *nonlinear_fun=&cnn::expr::rectify);
    ~PretagOutput();
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const IndexSeq &gold_seq);
    void build_output(const std::vector<cnn::expr::Expression> &expr_1,
                      const std::vector<cnn::expr::Expression> &expr_2,
                      IndexSeq &pred_out_seq) ;
};

struct CRFOutput
{
    Merge3Layer hidden_layer ;
    DenseLayer emit_layer ;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::LookupParameters *trans_score_lookup_param ;
    cnn::LookupParameters *init_score_lookup_param ;
    cnn::ComputationGraph *pcg ;
    size_t tag_num ;
    cnn::real dropout_rate ;
    NonLinearFunc *nonlinear_func ;
    CRFOutput(cnn::Model *m,
              unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
              unsigned hidden_dim,
              unsigned tag_num,
              cnn::real dropout_rate , 
              NonLinearFunc *nonlinear_func=&cnn::expr::rectify) ;
    ~CRFOutput() ;
    void new_graph(cnn::ComputationGraph &cg) ;
    cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
                        const std::vector<cnn::expr::Expression> &expr_cont2,
                        const IndexSeq &gold_seq) ;

    void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                      const std::vector<cnn::expr::Expression> &expr_cont2,
                      IndexSeq &pred_seq) ;
};


} // end of namespace slnn
#endif
