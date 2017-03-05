#ifndef MODELMODULE_HYPER_OUTPUT_LAYERS_H_
#define MODELMODULE_HYPER_OUTPUT_LAYERS_H_

#include <initializer_list>
#include <memory>
#include "layers.h"
#include "utils/typedeclaration.h"

namespace slnn{

/********* ABANDON 
 * Output Base 
 * output base and dirived class are old and should be **abandon**.
 *********/
struct OutputBase
{
    OutputBase(dynet::real dropout_rate, NonLinearFunc *nonlinear_func);
    virtual ~OutputBase() = 0 ;
    virtual void new_graph(dynet::ComputationGraph &cg) = 0 ;
    virtual dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const IndexSeq &gold_seq) = 0 ;
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        IndexSeq &pred_out_seq) = 0 ;
    dynet::real dropout_rate;
    NonLinearFunc *nonlinear_func;
};

struct SimpleOutput : public OutputBase
{
    Merge2Layer hidden_layer;
    DenseLayer output_layer;
    dynet::ComputationGraph *pcg;
    SimpleOutput(dynet::Model *m, unsigned input_dim1, unsigned input_dim2 ,
        unsigned hidden_dim, unsigned output_dim , 
        dynet::real dropout_rate=0.f, NonLinearFunc *nonlinear_func=&dynet::expr::rectify);
    virtual ~SimpleOutput();
    virtual void new_graph(dynet::ComputationGraph &cg);
    virtual dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2 ,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        IndexSeq &pred_out_seq);
};

struct PretagOutput : public OutputBase
{
    Merge3Layer hidden_layer;
    DenseLayer output_layer;
    dynet::LookupParameter tag_lookup_param ;
    dynet::Parameter TAG_SOS ;
    dynet::ComputationGraph *pcg;

    PretagOutput(dynet::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
        unsigned hidden_dim, unsigned output_dim , 
        dynet::real dropout_rate=0.f, NonLinearFunc *nonlinear_fun=&dynet::expr::rectify);
    virtual ~PretagOutput();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_1,
        const std::vector<dynet::expr::Expression> &expr_2,
        IndexSeq &pred_out_seq) ;
};

struct CRFOutput : public OutputBase
{
    Merge3Layer hidden_layer ;
    DenseLayer emit_layer ;
    dynet::LookupParameter tag_lookup_param ;
    dynet::LookupParameter trans_score_lookup_param ;
    dynet::LookupParameter init_score_lookup_param ;
    dynet::ComputationGraph *pcg ;
    size_t tag_num ;
    CRFOutput(dynet::Model *m,
        unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
        unsigned hidden_dim,
        unsigned tag_num,
        dynet::real dropout_rate , 
        NonLinearFunc *nonlinear_func=&dynet::expr::rectify) ;
    ~CRFOutput() ;
    void new_graph(dynet::ComputationGraph &cg) ;
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const IndexSeq &gold_seq) ;

    void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        IndexSeq &pred_seq) ;
};

/******************
 * Bare Output Base
 * should be supported in current and future.
 ******************/

struct BareOutputBase
{
    virtual ~BareOutputBase();
    virtual void new_graph(dynet::ComputationGraph &cg) = 0;
    dynet::expr::Expression
        build_output_loss(const std::vector<std::vector<dynet::expr::Expression> *> &input_expr_ptr_group_seq,
            const IndexSeq &gold_seq); 
    virtual dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &input_expr_seq,
            const IndexSeq &gold_seq) = 0;
    void build_output(const std::vector<std::vector<dynet::expr::Expression> *> &input_expr_ptr_group_seq,
        IndexSeq &predicted_seq);
    virtual void build_output(const std::vector<dynet::expr::Expression> &input_expr_seq,
        IndexSeq &predicted_seq) = 0 ;
private:
    void concate_input_expr_ptr_group(const std::vector<std::vector<dynet::expr::Expression> *> &input_expr_ptr_group_seq, 
            std::vector<dynet::expr::Expression> &concated_input_expr_cont);
};

/*************
 * Simple Bare output.
 * for general (no tag-transition limit)
 *************/

struct SimpleBareOutput : public BareOutputBase
{

public:
    SimpleBareOutput(dynet::Model *m, unsigned inputs_total_dim, unsigned output_dim);

public:
    void new_graph(dynet::ComputationGraph &cg) override;
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &input_expr_seq,
            const IndexSeq &gold_seq) override;
    void build_output(const std::vector<dynet::expr::Expression> &input_expr_seq,
        IndexSeq &predicted_seq) override;
protected:
    dynet::ComputationGraph *pcg;
    DenseLayer softmax_layer;
};

/******************
 * CRF Bare output.
 * for general.
 * *Attention:* this implementation is different from the previous CRFOutput implementation
 ******************/

class CrfBareOutput : public BareOutputBase
{

public:
    CrfBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim);

public:
    void new_graph(dynet::ComputationGraph &cg) override;
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression>& input_expr_seq,
            const std::vector<Index>& gold_output_seq) override;
    void build_output(const std::vector<dynet::expr::Expression>& input_expr_seq,
        std::vector<Index>& pred_output_seq) override;
private:
    // `to` keeps unchange when viterbi.
    unsigned flat_transition_index(unsigned from, unsigned to){ return from + tagdict_sz * to; }
protected:
    std::size_t tagdict_sz;
    DenseLayer state_score_layer;
    dynet::LookupParameter init_score_lookup_param;
    dynet::LookupParameter transition_score_lookup_param;
    dynet::ComputationGraph *pcg;
};


std::shared_ptr<BareOutputBase>
create_output_layer(const std::string& layer_type, dynet::Model* dynet_model, unsigned input_dim, unsigned output_dim);


// softmax layer

struct SoftmaxLayer
{
    dynet::ComputationGraph *pcg;
    DenseLayer output_layer;
    SoftmaxLayer(dynet::Model *m, unsigned input_dim, unsigned output_dim);
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &input_expr_cont,
            const IndexSeq &gold_seq);
    void build_output(const std::vector<dynet::expr::Expression> &input_expr_cont,
        IndexSeq &predicted_seq);
    Index build_output(dynet::expr::Expression input_expr);
    void get_output_distribution(dynet::expr::Expression input_expr, std::vector<dynet::real> &dist);
};


// Output Base Layer With Feature input

struct OutputBaseWithFeature
{
    OutputBaseWithFeature(dynet::real dropout_rate, NonLinearFunc *nonlinear_func);
    virtual ~OutputBaseWithFeature() = 0 ;
    virtual void new_graph(dynet::ComputationGraph &cg) = 0 ;
    virtual dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const std::vector<dynet::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq) = 0 ;
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        const std::vector<dynet::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq) = 0 ;
    dynet::real dropout_rate;
    NonLinearFunc *nonlinear_func;
};

struct SimpleOutputWithFeature : public OutputBaseWithFeature
{
    Merge3Layer hidden_layer;
    DenseLayer output_layer;
    dynet::ComputationGraph *pcg;
    SimpleOutputWithFeature(dynet::Model *m, unsigned input_dim1, unsigned input_dim2, unsigned feature_dim,
        unsigned hidden_dim, unsigned output_dim,
        dynet::real dropout_rate=0.f, NonLinearFunc *nonlinear_func=&dynet::expr::rectify);
    virtual ~SimpleOutputWithFeature();
    virtual void new_graph(dynet::ComputationGraph &cg);
    virtual dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2 ,
            const std::vector<dynet::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        const std::vector<dynet::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq);
};

struct PretagOutputWithFeature : public OutputBaseWithFeature
{
    Merge4Layer hidden_layer;
    DenseLayer output_layer;
    dynet::LookupParameter tag_lookup_param ;
    dynet::Parameter TAG_SOS ;
    dynet::ComputationGraph *pcg;

    PretagOutputWithFeature(dynet::Model *m, unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2, 
        unsigned feature_dim,
        unsigned hidden_dim, unsigned output_dim , 
        dynet::real dropout_rate=0.f, NonLinearFunc *nonlinear_fun=&dynet::expr::rectify);
    virtual ~PretagOutputWithFeature();
    void new_graph(dynet::ComputationGraph &cg);
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const std::vector<dynet::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq);
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_1,
        const std::vector<dynet::expr::Expression> &expr_2,
        const std::vector<dynet::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq) ;
};

struct CRFOutputWithFeature : public  OutputBaseWithFeature
{
    Merge4Layer hidden_layer ;
    DenseLayer emit_layer ;
    dynet::LookupParameter tag_lookup_param ;
    dynet::LookupParameter trans_score_lookup_param ;
    dynet::LookupParameter init_score_lookup_param ;
    dynet::ComputationGraph *pcg ;
    size_t tag_num ;
    CRFOutputWithFeature(dynet::Model *m,
        unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
        unsigned feature_dim,
        unsigned hidden_dim,
        unsigned tag_num,
        dynet::real dropout_rate , 
        NonLinearFunc *nonlinear_func=&dynet::expr::rectify) ;
    ~CRFOutputWithFeature() ;
    void new_graph(dynet::ComputationGraph &cg) ;
    dynet::expr::Expression
        build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
            const std::vector<dynet::expr::Expression> &expr_cont2,
            const std::vector<dynet::expr::Expression> &feature_expr_cont,
            const IndexSeq &gold_seq) ;

    void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        const std::vector<dynet::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_seq) ;
};


/*  inline function implementation */

/***** simple output *****/
inline
void SimpleOutput::new_graph(dynet::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg);
    output_layer.new_graph(cg);
    pcg = &cg;
}

inline
Expression SimpleOutput::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2 , const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size();
    std::vector<dynet::expr::Expression> loss_cont(len);
    for (size_t i = 0; i < len; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression dropout_expr = dynet::expr::dropout(nonlinear_expr, dropout_rate);
        dynet::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = dynet::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
    }
    return dynet::expr::sum(loss_cont);
}

inline
void SimpleOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i]);
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out[i] = idx_of_max_prob;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}


/***** pretag output ******/
inline
void PretagOutput::new_graph(dynet::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg) ;
    output_layer.new_graph(cg) ;
    pcg = &cg ;
}

inline
dynet::expr::Expression
PretagOutput::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    std::vector<dynet::expr::Expression> loss_cont(len);
    dynet::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i] , pretag_exp);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression dropout_expr = dynet::expr::dropout(nonlinear_expr, dropout_rate);
        dynet::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = dynet::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
        pretag_exp = lookup(*pcg, tag_lookup_param, gold_seq.at(i)) ;
    }
    return dynet::expr::sum(loss_cont);
}

inline
void PretagOutput::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    dynet::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1[i], expr_cont2[i], pretag_exp);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_max_prob = std::distance(dist.cbegin(), std::max_element(dist.cbegin(), dist.cend())) ;
        tmp_pred[i] = id_of_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_max_prob) ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

/****** crf output *******/
inline
void CRFOutput::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg ;
    hidden_layer.new_graph(cg) ;
    emit_layer.new_graph(cg) ;
}

inline 
void BareOutputBase::concate_input_expr_ptr_group(const std::vector<std::vector<dynet::expr::Expression>*> &input_expr_ptr_group_seq,
    std::vector<dynet::expr::Expression> &concated_input_expr_cont)
{
    // col-based concatenate
    // every row is one feature seq,
    // every col is the features of every feature seq at this column
    size_t nr_feature_variety = input_expr_ptr_group_seq.size();
    //assert(nr_feature_variety > 0);
    size_t feature_seq_len = input_expr_ptr_group_seq[0]->size();
    std::vector<dynet::expr::Expression> tmp_concated_expr(feature_seq_len);
    std::vector<dynet::expr::Expression> feature_group(nr_feature_variety);
    for( size_t j = 0; j < feature_seq_len; ++j )
    {
        for( size_t fi = 0; fi < nr_feature_variety; ++fi )
        {
            feature_group.at(fi) = input_expr_ptr_group_seq[fi]->at(j);
        }
        tmp_concated_expr.at(j) = dynet::expr::concatenate(feature_group);
    }
    swap(concated_input_expr_cont, tmp_concated_expr);
}

inline
dynet::expr::Expression
BareOutputBase::build_output_loss(const std::vector<std::vector<dynet::expr::Expression> *> &input_expr_ptr_group_seq,
    const IndexSeq &gold_seq)
{
    std::vector<dynet::expr::Expression> merged_expr_cont;
    concate_input_expr_ptr_group(input_expr_ptr_group_seq, merged_expr_cont);
    return build_output_loss(merged_expr_cont, gold_seq);
}
inline
void BareOutputBase::build_output(const std::vector<std::vector<dynet::expr::Expression> *> &input_expr_ptr_group_seq,
    IndexSeq &predicted_seq)
{
    std::vector<dynet::expr::Expression> merged_expr_cont;
    concate_input_expr_ptr_group(input_expr_ptr_group_seq, merged_expr_cont);
    return build_output(merged_expr_cont, predicted_seq);
}

/* Sample BareOutput Base */

/* Bare Output Base */
inline
void SimpleBareOutput::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg;
    softmax_layer.new_graph(cg);
}

inline
dynet::expr::Expression
SimpleBareOutput::build_output_loss(const std::vector<dynet::expr::Expression> &input_expr_seq,
    const IndexSeq &gold_seq) 
{
    size_t seq_len = input_expr_seq.size();
    std::vector<dynet::expr::Expression> loss_cont(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        dynet::expr::Expression dist_expr = softmax_layer.build_graph(input_expr_seq[i]);
        loss_cont[i] = dynet::expr::pickneglogsoftmax(dist_expr, gold_seq[i]);
    }
    return dynet::expr::sum(loss_cont);
}

inline
void SimpleBareOutput::build_output(const std::vector<dynet::expr::Expression> &input_expr_seq,
    IndexSeq &predicted_seq)
{
    using std::swap;
    size_t seq_len = input_expr_seq.size();
    IndexSeq tmp_pred_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        dynet::expr::Expression dest_expr = softmax_layer.build_graph(input_expr_seq[i]);
        std::vector<dynet::real> dist = dynet::as_vector(pcg->get_value(dest_expr));
        Index id_of_max_prob = std::distance(dist.begin(), std::max_element(dist.begin(), dist.end()) );
        tmp_pred_seq[i] = id_of_max_prob;
    }
    swap(predicted_seq, tmp_pred_seq);
}

/************
 * crf bare output class.
 **/

inline
void CrfBareOutput::new_graph(dynet::ComputationGraph& cg)
{
    state_score_layer.new_graph(cg);
    pcg = &cg;
}


/******* Softmax layer **********/

inline
void SoftmaxLayer::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg;
    output_layer.new_graph(cg);
}

inline
dynet::expr::Expression
SoftmaxLayer::build_output_loss(const std::vector<dynet::expr::Expression> &input_expr_cont,
    const IndexSeq &gold_seq)
{
    unsigned sz = input_expr_cont.size();
    std::vector<dynet::expr::Expression> loss_expr_cont(sz);
    for( unsigned i = 0 ; i < sz ; ++i )
    {
        dynet::expr::Expression output_expr = output_layer.build_graph(input_expr_cont.at(i));
        loss_expr_cont.at(i) = dynet::expr::pickneglogsoftmax(output_expr, gold_seq.at(i));
    }
    return dynet::expr::sum(loss_expr_cont);
}

inline
void SoftmaxLayer::build_output(const std::vector<dynet::expr::Expression> &input_expr_cont,
    IndexSeq &predicted_seq)
{
    size_t len = input_expr_cont.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        dynet::expr::Expression out_expr = output_layer.build_graph(input_expr_cont.at(i));
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out.at(i) = idx_of_max_prob;
    }
    std::swap(predicted_seq, tmp_pred_out);
}

inline
Index SoftmaxLayer::build_output(dynet::expr::Expression input_expr)
{
    std::vector<dynet::real> out_probs;
    get_output_distribution(input_expr, out_probs);
    return std::distance(out_probs.cbegin(), std::max_element(out_probs.cbegin(), out_probs.cend()));
}

inline
void SoftmaxLayer::get_output_distribution(dynet::expr::Expression input_expr, std::vector<dynet::real> &dist)
{
    dynet::expr::Expression out_expr = output_layer.build_graph(input_expr);
    dist = dynet::as_vector(pcg->get_value(out_expr));
}


/******* simple output with feature ********/

inline
void SimpleOutputWithFeature::new_graph(dynet::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg);
    output_layer.new_graph(cg);
    pcg = &cg;
}

inline
Expression SimpleOutputWithFeature::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2 , 
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size();
    std::vector<dynet::expr::Expression> loss_cont(len);
    for (size_t i = 0; i < len; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i));
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression dropout_expr = dynet::expr::dropout(nonlinear_expr, dropout_rate);
        dynet::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = dynet::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
    }
    return dynet::expr::sum(loss_cont);
}

inline
void SimpleOutputWithFeature::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_out_seq)
{
    size_t len = expr_cont1.size();
    std::vector<Index> tmp_pred_out(len);
    for (size_t i = 0; i < len; ++i)
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i));
        dynet::expr::Expression nonlinear_expr = nonlinear_func(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        Index idx_of_max_prob = std::distance(out_probs.cbegin(),
            std::max_element(out_probs.cbegin(), out_probs.cend()));
        tmp_pred_out[i] = idx_of_max_prob;
    }
    std::swap(pred_out_seq, tmp_pred_out);
}

/* pretag output with feature  */

inline 
void PretagOutputWithFeature::new_graph(dynet::ComputationGraph &cg)
{
    hidden_layer.new_graph(cg) ;
    output_layer.new_graph(cg) ;
    pcg = &cg ;
}


inline
dynet::expr::Expression
PretagOutputWithFeature::build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    const IndexSeq &gold_seq)
{
    size_t len = expr_cont1.size() ;
    std::vector<dynet::expr::Expression> loss_cont(len);
    dynet::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i),
            pretag_exp);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression dropout_expr = dynet::expr::dropout(nonlinear_expr, dropout_rate);
        dynet::expr::Expression out_expr = output_layer.build_graph(dropout_expr);
        loss_cont[i] = dynet::expr::pickneglogsoftmax(out_expr, gold_seq.at(i));
        pretag_exp = lookup(*pcg, tag_lookup_param, gold_seq.at(i)) ;
    }
    return dynet::expr::sum(loss_cont);
}

inline
void PretagOutputWithFeature::build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
    const std::vector<dynet::expr::Expression> &expr_cont2,
    const std::vector<dynet::expr::Expression> &feature_expr_cont,
    IndexSeq &pred_seq)
{
    size_t len = expr_cont1.size() ;
    IndexSeq tmp_pred(len) ;
    dynet::expr::Expression pretag_exp = parameter(*pcg, TAG_SOS) ;
    for( size_t i = 0; i < len; ++i )
    {
        dynet::expr::Expression merge_out_expr = hidden_layer.build_graph(expr_cont1.at(i), expr_cont2.at(i), feature_expr_cont.at(i), pretag_exp);
        dynet::expr::Expression nonlinear_expr = (*nonlinear_func)(merge_out_expr);
        dynet::expr::Expression out_expr = output_layer.build_graph(nonlinear_expr);
        std::vector<dynet::real> dist = as_vector(pcg->get_value(out_expr)) ;
        Index id_of_max_prob = std::distance(dist.cbegin(), std::max_element(dist.cbegin(), dist.cend())) ;
        tmp_pred[i] = id_of_max_prob ;
        pretag_exp = lookup(*pcg, tag_lookup_param,id_of_max_prob) ;
    }
    std::swap(pred_seq, tmp_pred) ;
}

/* CRF output with feature */
inline
void CRFOutputWithFeature::new_graph(dynet::ComputationGraph &cg)
{
    pcg = &cg ;
    hidden_layer.new_graph(cg) ;
    emit_layer.new_graph(cg) ;
}

}

#endif
