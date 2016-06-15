#ifndef POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_H_
#define POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_H_

#include <boost/program_options.hpp>

#include "cnn/cnn.h"

#include "modelmodule/context_feature.h"
#include "modelmodule/context_feature_extractor.h"
#include "modelmodule/context_feature_layer.h"
#include "postagger/postagger_module/pos_feature.h"
#include "postagger/postagger_module/pos_feature_extractor.h"
#include "postagger/postagger_module/pos_feature_layer.h"

namespace slnn{

class Input1MLPModel
{
    friend class boost::serialization::access;
public :
    static const std::string UNK_STR;
    static const std::string StrOfReplaceNumber ;
    static const size_t LenStrOfRepalceNumber ;
    static constexpr size_t PostaggerContextSize = 4 ;
    using POSContextFeature = ContextFeature<PostaggerContextSize>;

public:
    Input1MLPModel();
    virtual ~Input1MLPModel();
    Input1MLPModel(const Input1MLPModel &) = delete;
    Input1MLPModel & operator=(const Input1MLPModel&) = delete;

    // dict and model interface
    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    cnn::Dict& get_word_dict(){ return word_dict ;  } 
    cnn::Dict& get_postag_dict(){ return postag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } 


    // dirived class should override
    virtual void set_model_param(const boost::program_options::variables_map &var_map) ;
    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    // process input and build dict
    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
        IndexSeq &index_sent, IndexSeq &index_postag_seq, 
        POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for annotated data

    void input_seq2index_seq(const Seq &sent, 
        IndexSeq &index_sent, 
        POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for input data

    void replace_word_with_unk(const IndexSeq &sent, 
        const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
        IndexSeq &replaced_sent,
        POSContextFeature::ContextFeatureIndexGroupSeq &context_replaced_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq);
    
    // output translate
    void postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq);

    // training and predict
    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq)  = 0 ;
    virtual void predict(cnn::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) = 0 ;

    // print
    std::string get_mlp_hidden_layer_dim_info();

protected:
    cnn::Model *m;

    cnn::Dict word_dict;
    cnn::Dict postag_dict;
    DictWrapper word_dict_wrapper;
public:
    POSFeature pos_feature; // also as parameters
    POSContextFeature context_feature;

public :
    unsigned word_embedding_dim,
        word_dict_size,
        input_dim,
        output_dim;
    std::vector<unsigned> mlp_hidden_dim_list;
    cnn::real dropout_rate;
};

using POSContextFeature = Input1MLPModel::POSContextFeature;

/*********** inline funtion implemantation **********/

// for short and frequently called functions

void Input1MLPModel::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
    pos_feature.set_replace_feature_with_unk_threshold(freq_threshold, prob_threshold);
}

bool Input1MLPModel::is_dict_frozen()
{
    return (word_dict.is_frozen() && postag_dict.is_frozen() && pos_feature.is_dict_frozen());
}

void Input1MLPModel::freeze_dict()
{
    word_dict_wrapper.Freeze();
    postag_dict.Freeze();
    word_dict_wrapper.SetUnk(UNK_STR);
    pos_feature.freeze_dict();
}


inline
void Input1MLPModel::replace_word_with_unk(const IndexSeq &sent,
    const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
    const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
    IndexSeq &replaced_sent, 
    POSContextFeature::ContextFeatureIndexGroupSeq &replaced_context_feature_gp_seq,
    POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_rep_sent(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_rep_sent[i] = word_dict_wrapper.ConvertProbability(sent[i]);
    }
    swap(replaced_sent, tmp_rep_sent);
    pos_feature.do_repalce_feature_with_unk_in_copy(feature_gp_seq, replaced_feature_gp_seq);
    context_feature.replace_feature_index_group_seq_with_unk(context_feature_gp_seq, replaced_context_feature_gp_seq);
}

inline
void Input1MLPModel::postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq)
{
    size_t seq_len = postag_index_seq.size();
    Seq tmp_str_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_str_seq[i] = postag_dict.Convert(postag_index_seq[i]);
    }
    swap(postag_str_seq, tmp_str_seq);
}

} // end of namespace slnn


#endif