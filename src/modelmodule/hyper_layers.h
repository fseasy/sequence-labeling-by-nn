#ifndef HYPER_LAYERS_H_INCLUDED_
#define HYPER_LAYERS_H_INCLUDED_

#include <initializer_list>
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

struct Input1WithFeature
{
    cnn::LookupParameters *word_lookup_param;
    Merge2Layer m2_layer;
    cnn::ComputationGraph *pcg;
    NonLinearFunc *nonlinear_func;
    Input1WithFeature(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim,
                      unsigned feature_embedding_dim, unsigned merge_out_dim,
                      NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    void new_graph(cnn::ComputationGraph &cg);
    void build_inputs(const IndexSeq &sent , const std::vector<cnn::expr::Expression> &feature_exprs,
                      std::vector<cnn::expr::Expression> &input_exprs);
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

struct BareInput1
{
    cnn::LookupParameters *word_lookup_param;
    cnn::ComputationGraph *pcg;
    unsigned nr_exprs;
    std::vector<cnn::expr::Expression> exprs;
    BareInput1(cnn::Model *m, unsigned vocabulary_size, unsigned word_embedding_dim, unsigned nr_extra_feature_exprs);
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
    build_input(Index word_idx, const std::vector<cnn::expr::Expression> &extra_feature_exprs);
    void build_inputs(const IndexSeq &sent, const std::vector<std::vector<cnn::expr::Expression>> &extra_feature_exprs_seq,
        std::vector<cnn::expr::Expression> &input_exprs);
};


struct OutputBase
{
    OutputBase(cnn::real dropout_rate, NonLinearFunc *nonlinear_func);
    virtual ~OutputBase() = 0 ;
    virtual void new_graph(cnn::ComputationGraph &cg) = 0 ;
    virtual cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
                      const std::vector<cnn::expr::Expression> &expr_cont2,
                      const IndexSeq &gold_seq) = 0 ;
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                              const std::vector<cnn::expr::Expression> &expr_cont2,
                              IndexSeq &pred_out_seq) = 0 ;
    cnn::real dropout_rate;
    NonLinearFunc *nonlinear_func;
};

struct SimpleOutput : public OutputBase
{
    Merge2Layer hidden_layer;
    DenseLayer output_layer;
    cnn::ComputationGraph *pcg;
    SimpleOutput(cnn::Model *m, unsigned input_dim1, unsigned input_dim2 ,
        unsigned hidden_dim, unsigned output_dim , 
        cnn::real dropout_rate=0.f, NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    virtual ~SimpleOutput();
    virtual void new_graph(cnn::ComputationGraph &cg);
    virtual cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
           const std::vector<cnn::expr::Expression> &expr_cont2 ,
           const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        IndexSeq &pred_out_seq);
};

struct PretagOutput : public OutputBase
{
    Merge3Layer hidden_layer;
    DenseLayer output_layer;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::Parameters *TAG_SOS ;
    cnn::ComputationGraph *pcg;

    PretagOutput(cnn::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
                 unsigned hidden_dim, unsigned output_dim , 
                 cnn::real dropout_rate=0.f, NonLinearFunc *nonlinear_fun=&cnn::expr::rectify);
    virtual ~PretagOutput();
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_1,
                      const std::vector<cnn::expr::Expression> &expr_2,
                      IndexSeq &pred_out_seq) ;
};

struct CRFOutput : public OutputBase
{
    Merge3Layer hidden_layer ;
    DenseLayer emit_layer ;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::LookupParameters *trans_score_lookup_param ;
    cnn::LookupParameters *init_score_lookup_param ;
    cnn::ComputationGraph *pcg ;
    size_t tag_num ;
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

// softmax layer

struct SoftmaxLayer
{
    cnn::ComputationGraph *pcg;
    DenseLayer output_layer;
    SoftmaxLayer(cnn::Model *m, unsigned input_dim, unsigned output_dim);
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
        build_output_loss(const std::vector<cnn::expr::Expression> &input_expr_cont,
            const IndexSeq &gold_seq);
    void build_output(const std::vector<cnn::expr::Expression> &input_expr_cont,
        IndexSeq &predicted_seq);
};


// Output Base Layer With Feature input

struct OutputBaseWithFeature
{
    OutputBaseWithFeature(cnn::real dropout_rate, NonLinearFunc *nonlinear_func);
    virtual ~OutputBaseWithFeature() = 0 ;
    virtual void new_graph(cnn::ComputationGraph &cg) = 0 ;
    virtual cnn::expr::Expression
    build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const std::vector<cnn::expr::Expression> &feature_expr_cont,
        const IndexSeq &gold_seq) = 0 ;
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const std::vector<cnn::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq) = 0 ;
    cnn::real dropout_rate;
    NonLinearFunc *nonlinear_func;
};

struct SimpleOutputWithFeature : public OutputBaseWithFeature
{
    Merge3Layer hidden_layer;
    DenseLayer output_layer;
    cnn::ComputationGraph *pcg;
    SimpleOutputWithFeature(cnn::Model *m, unsigned input_dim1, unsigned input_dim2, unsigned feature_dim,
        unsigned hidden_dim, unsigned output_dim,
        cnn::real dropout_rate=0.f, NonLinearFunc *nonlinear_func=&cnn::expr::rectify);
    virtual ~SimpleOutputWithFeature();
    virtual void new_graph(cnn::ComputationGraph &cg);
    virtual cnn::expr::Expression
        build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
            const std::vector<cnn::expr::Expression> &expr_cont2 ,
            const std::vector<cnn::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const std::vector<cnn::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq);
};

struct PretagOutputWithFeature : public OutputBaseWithFeature
{
    Merge4Layer hidden_layer;
    DenseLayer output_layer;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::Parameters *TAG_SOS ;
    cnn::ComputationGraph *pcg;

    PretagOutputWithFeature(cnn::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2, 
        unsigned feature_dim,
        unsigned hidden_dim, unsigned output_dim , 
        cnn::real dropout_rate=0.f, NonLinearFunc *nonlinear_fun=&cnn::expr::rectify);
    virtual ~PretagOutputWithFeature();
    void new_graph(cnn::ComputationGraph &cg);
    cnn::expr::Expression
        build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
            const std::vector<cnn::expr::Expression> &expr_cont2,
            const std::vector<cnn::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<cnn::expr::Expression> &expr_1,
        const std::vector<cnn::expr::Expression> &expr_2,
        const std::vector<cnn::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq) ;
};

struct CRFOutputWithFeature : public  OutputBaseWithFeature
{
    Merge4Layer hidden_layer ;
    DenseLayer emit_layer ;
    cnn::LookupParameters *tag_lookup_param ;
    cnn::LookupParameters *trans_score_lookup_param ;
    cnn::LookupParameters *init_score_lookup_param ;
    cnn::ComputationGraph *pcg ;
    size_t tag_num ;
    CRFOutputWithFeature(cnn::Model *m,
        unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
        unsigned feature_dim,
        unsigned hidden_dim,
        unsigned tag_num,
        cnn::real dropout_rate , 
        NonLinearFunc *nonlinear_func=&cnn::expr::rectify) ;
    ~CRFOutputWithFeature() ;
    void new_graph(cnn::ComputationGraph &cg) ;
    cnn::expr::Expression
        build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
            const std::vector<cnn::expr::Expression> &expr_cont2,
            const std::vector<cnn::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq) ;

    void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
        const std::vector<cnn::expr::Expression> &expr_cont2,
        const std::vector<cnn::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_seq) ;
};

/******* inline function implementation ******/

/****** input1 ******/
inline
void Input1::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
}

inline
void Input1::build_inputs(const IndexSeq &sent, std::vector<cnn::expr::Expression> &inputs_exprs)
{
    if (nullptr == pcg) throw std::runtime_error("cg should be set .");
    size_t sent_len = sent.size();
    std::vector<cnn::expr::Expression> tmp_inputs(sent_len);
    for (size_t i = 0; i < sent_len; ++i)
    {
        tmp_inputs[i] = lookup(*pcg, word_lookup_param, sent[i]);
    }
    std::swap(inputs_exprs, tmp_inputs);
}

/******* input with feature ******/
inline
void Input1WithFeature::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m2_layer.new_graph(cg);
}

inline
void Input1WithFeature::build_inputs(const IndexSeq &sent, 
    const std::vector<cnn::expr::Expression> &features_exprs,
    std::vector<cnn::expr::Expression> &inputs_exprs)
{
    using std::swap;
    if (nullptr == pcg) throw std::runtime_error("cg should be set .");
    size_t sent_len = sent.size();
    std::vector<cnn::expr::Expression> tmp_inputs(sent_len);
    for (size_t i = 0; i < sent_len; ++i)
    {
        cnn::expr::Expression word_expr = lookup(*pcg, word_lookup_param, sent[i]);
        const cnn::expr::Expression &feature_expr = features_exprs[i];
        cnn::expr::Expression merge_expr = m2_layer.build_graph(word_expr, feature_expr);
        tmp_inputs[i] = (*nonlinear_func)(merge_expr);
    }
    std::swap(inputs_exprs, tmp_inputs);
}
/******* input 2d  *******/
inline
void Input2D::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m2_layer.new_graph(cg);
}

inline
void Input2D::build_inputs(const IndexSeq &seq1, const IndexSeq &seq2, std::vector<cnn::expr::Expression> &inputs_exprs )
{
    size_t seq_len = seq1.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression expr1 = lookup(*pcg, dynamic_lookup_param1, seq1.at(i));
        cnn::expr::Expression expr2 = lookup(*pcg, dynamic_lookup_param2, seq2.at(i));
        cnn::expr::Expression linear_merge_expr = m2_layer.build_graph(expr1, expr2);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(linear_merge_expr);
        tmp_inputs[i] = nonlinear_expr;
    }
    std::swap(inputs_exprs, tmp_inputs);
}

/******** input 2 ***********/
inline
void Input2::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m2_layer.new_graph(cg);
}

inline
void Input2::build_inputs(const IndexSeq &dynamic_seq, const IndexSeq &fixed_seq, std::vector<cnn::expr::Expression> &inputs_exprs)
{
    size_t seq_len = dynamic_seq.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression expr1 = lookup(*pcg, dynamic_lookup_param, dynamic_seq.at(i));
        cnn::expr::Expression expr2 = lookup(*pcg, fixed_lookup_param, fixed_seq.at(i));
        cnn::expr::Expression linear_merge_expr = m2_layer.build_graph(expr1, expr2);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(linear_merge_expr);
        tmp_inputs[i] = nonlinear_expr;
    }
    std::swap(inputs_exprs, tmp_inputs);
}
/******* input 3  ********/
inline
void Input3::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    m3_layer.new_graph(cg);
}

inline
void Input3::build_inputs(const IndexSeq &dseq1, const IndexSeq &dseq2, const IndexSeq &fseq,
    std::vector<cnn::expr::Expression> &inputs_exprs)
{
    size_t seq_len = dseq1.size();
    std::vector<cnn::expr::Expression> tmp_inputs(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        cnn::expr::Expression dexpr1 = lookup(*pcg, dynamic_lookup_param1, dseq1.at(i));
        cnn::expr::Expression dexpr2 = lookup(*pcg, dynamic_lookup_param2, dseq2.at(i));
        cnn::expr::Expression fexpr = const_lookup(*pcg, fixed_lookup_param, fseq.at(i));
        cnn::expr::Expression linear_merge_expr = m3_layer.build_graph(dexpr1, dexpr2, fexpr);
        tmp_inputs[i] = nonlinear_func(linear_merge_expr);
    }
    std::swap(inputs_exprs, tmp_inputs);
}
/******* bare input1  *******/
inline
void BareInput1::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
}

inline
cnn::expr::Expression
BareInput1::build_input(Index word_idx, const std::vector<cnn::expr::Expression> &extra_feature_exprs)
{
    exprs.at(0) = lookup(*pcg, word_lookup_param, word_idx);
    for( unsigned i = 1 ; i < nr_exprs; ++i )
    {
        exprs.at(i) = extra_feature_exprs.at(i-1);
    }
    return cnn::expr::concatenate(exprs);
}

inline 
void BareInput1::build_inputs(const IndexSeq &sent, const std::vector<std::vector<cnn::expr::Expression>> &extra_feature_exprs_seq,
    std::vector<cnn::expr::Expression> &input_exprs)
{
    using std::swap;
    unsigned seq_len = sent.size();
    std::vector<cnn::expr::Expression> concat_result(seq_len);
    for( unsigned i = 0; i < seq_len ; ++i )
    {
        concat_result[i] = build_input(sent[i], extra_feature_exprs_seq[i]);
    }
    swap(concat_result, input_exprs);
}

/***** simple output *****/
inline
void SimpleOutput::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg);
    output_layer.new_graph(cg);
    pcg = &cg;
}

inline
Expression SimpleOutput::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2 , const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size();
    std::vector<cnn::expr::Expression> loss_cont(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression dropout_expr = cnn::expr::dropout(nonlinear_expr, dropout_rate);
        cnn::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
    }
    return cnn::expr::sum(loss_cont);
}

inline
void SimpleOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> out_probs = cnn::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out[i] = idx_of_max_prob;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}


/***** pretag output ******/
inline
void PretagOutput::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg) ;
    output_layer.new_graph(cg) ;
    pcg = &cg ;
}

inline
cnn::expr::Expression
PretagOutput::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    std::vector<cnn::expr::Expression> loss_cont(len);
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i] , pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression dropout_expr = cnn::expr::dropout(nonlinear_expr, dropout_rate);
        cnn::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
        pretag_exp = lookup(*pcg, tag_lookup_param, gold_seq.at(i)) ;
    }
    return cnn::expr::sum(loss_cont);
}

inline
void PretagOutput::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_max_prob = std::distance(dist.cbegin(), std::max_element(dist.cbegin(), dist.cend())) ;
        tmp_pred[i] = id_of_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_max_prob) ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

/****** crf output *******/
inline
void CRFOutput::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg ;
    hidden_layer.new_graph(cg) ;
    emit_layer.new_graph(cg) ;
}


/******* Softmax layer **********/

inline
void SoftmaxLayer::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
    output_layer.new_graph(cg);
}

inline
cnn::expr::Expression
SoftmaxLayer::build_output_loss(const std::vector<cnn::expr::Expression> &input_expr_cont,
    const IndexSeq &gold_seq)
{
    unsigned sz = input_expr_cont.size();
    std::vector<cnn::expr::Expression> loss_expr_cont(sz);
    for( unsigned i = 0 ; i < sz ; ++i )
    {
        cnn::expr::Expression output_expr = output_layer.build_graph(input_expr_cont.at(i));
        loss_expr_cont.at(i) = cnn::expr::pickneglogsoftmax(output_expr, gold_seq.at(i));
    }
    return cnn::expr::sum(loss_expr_cont);
}

inline
void SoftmaxLayer::build_output(const std::vector<cnn::expr::Expression> &input_expr_cont,
    IndexSeq &predicted_seq)
{
    size_t len = input_expr_cont.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression out_expr = output_layer.build_graph(input_expr_cont.at(i));
        std::vector<cnn::real> out_probs = cnn::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out.at(i) = idx_of_max_prob;
    }
    std::swap(predicted_seq, tmp_pred_out);
}

/******* simple output with feature ********/

inline
void SimpleOutputWithFeature::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg);
    output_layer.new_graph(cg);
    pcg = &cg;
}

inline
Expression SimpleOutputWithFeature::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2 , 
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size();
    std::vector<cnn::expr::Expression> loss_cont(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i));
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression dropout_expr = cnn::expr::dropout(nonlinear_expr, dropout_rate);
        cnn::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
    }
    return cnn::expr::sum(loss_cont);
}

inline
void SimpleOutputWithFeature::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i));
        cnn::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> out_probs = cnn::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out[i] = idx_of_max_prob;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}

/* pretag output with feature  */

inline 
void PretagOutputWithFeature::new_graph(cnn::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg) ;
    output_layer.new_graph(cg) ;
    pcg = &cg ;
}


inline
cnn::expr::Expression
PretagOutputWithFeature::build_output_loss(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    std::vector<cnn::expr::Expression> loss_cont(len);
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i),
            pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression dropout_expr = cnn::expr::dropout(nonlinear_expr, dropout_rate);
        cnn::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = cnn::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
        pretag_exp = lookup(*pcg, tag_lookup_param, gold_seq.at(i)) ;
    }
    return cnn::expr::sum(loss_cont);
}

inline
void PretagOutputWithFeature::build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
    const std::vector<cnn::expr::Expression> &expr_cont2,
    const std::vector<cnn::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    cnn::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        cnn::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i), pretag_exp);
        cnn::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        cnn::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<cnn::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_max_prob = std::distance(dist.cbegin(), std::max_element(dist.cbegin(), dist.cend())) ;
        tmp_pred[i] = id_of_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_max_prob) ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

/* CRF output with feature */
inline
void CRFOutputWithFeature::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg ;
    hidden_layer.new_graph(cg) ;
    emit_layer.new_graph(cg) ;
}

} // end of namespace slnn
#endif
