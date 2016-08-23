#ifndef POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_NO_FEATURE_H_
#define POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_NO_FEATURE_H_

#include <boost/program_options.hpp>

#include "cnn/cnn.h"

#include "modelmodule/context_feature.h"
#include "modelmodule/context_feature_layer.h"
#include "postagger/postagger_module/pos_reader.h"
namespace slnn{

class Input1MLPModelNoFeature
{
    friend class boost::serialization::access;
public :
    static const std::string UNK_STR;
    static const std::string StrOfReplaceNumber ;
    static const size_t LenStrOfRepalceNumber ;

public:
    Input1MLPModelNoFeature();
    virtual ~Input1MLPModelNoFeature();
    Input1MLPModelNoFeature(const Input1MLPModelNoFeature &) = delete;
    Input1MLPModelNoFeature & operator=(const Input1MLPModelNoFeature&) = delete;

    // dict and model interface
    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    cnn::Dict& get_word_dict(){ return word_dict ;  } 
    cnn::Dict& get_postag_dict(){ return postag_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } 


    // dirived class should override
    virtual void set_model_param_from_outer(const boost::program_options::variables_map &var_map) ;
    virtual void set_model_param_from_inner();

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    // process input and build dict
    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
        IndexSeq &index_sent, IndexSeq &index_postag_seq, 
        ContextFeatureDataSeq &context_feature_gp_seq); // for annotated data

    void input_seq2index_seq(const Seq &sent, 
        IndexSeq &index_sent, 
        ContextFeatureDataSeq &context_feature_gp_seq); // for input data

    void replace_word_with_unk(const IndexSeq &sent, 
        const ContextFeatureDataSeq &context_feature_gp_seq,
        IndexSeq &replaced_sent,
        ContextFeatureDataSeq &context_replaced_feature_gp_seq);
    
    // output translate
    void postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq);

    // training and predict
    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const ContextFeatureDataSeq &context_feature_gp_seq,
        const IndexSeq &gold_seq)  = 0 ;
    virtual void predict(cnn::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const ContextFeatureDataSeq &context_feature_gp_seq,
        IndexSeq &pred_seq) = 0 ;

    // print
    std::string get_mlp_hidden_layer_dim_info();

    // serialization
    template <typename Archive>
    void save(Archive &ar, unsigned version) const ;
    template <typename Archive>
    void load(Archive &ar, unsigned version);
    template <typename Archive>
    void serialize(Archive &ar, unsigned version);

protected:
    cnn::Model *m;

    cnn::Dict word_dict;
    cnn::Dict postag_dict;
    DictWrapper word_dict_wrapper;
public:
    ContextFeature context_feature;

public :
    unsigned word_embedding_dim,
        word_dict_size,
        input_dim,
        output_dim;
    std::vector<unsigned> mlp_hidden_dim_list;
    cnn::real dropout_rate;
};

/*********** inline funtion implemantation **********/

// for short and frequently called functions
inline
void Input1MLPModelNoFeature::set_replace_threshold(int freq_threshold, float prob_threshold)
{
    word_dict_wrapper.set_threshold(freq_threshold, prob_threshold);
}

inline
bool Input1MLPModelNoFeature::is_dict_frozen()
{
    return (word_dict.is_frozen() && postag_dict.is_frozen());
}

inline
void Input1MLPModelNoFeature::freeze_dict()
{
    word_dict_wrapper.Freeze();
    postag_dict.Freeze();
    word_dict_wrapper.SetUnk(UNK_STR);
}


inline
void Input1MLPModelNoFeature::replace_word_with_unk(const IndexSeq &sent,
    const ContextFeatureDataSeq &context_feature_gp_seq,
    IndexSeq &replaced_sent, 
    ContextFeatureDataSeq &replaced_context_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_rep_sent(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_rep_sent[i] = word_dict_wrapper.ConvertProbability(sent[i]);
    }
    swap(replaced_sent, tmp_rep_sent);
    context_feature.random_replace_with_unk(context_feature_gp_seq, replaced_context_feature_gp_seq);
}

inline
void Input1MLPModelNoFeature::postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq)
{
    size_t seq_len = postag_index_seq.size();
    Seq tmp_str_seq(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        tmp_str_seq[i] = postag_dict.Convert(postag_index_seq[i]);
    }
    swap(postag_str_seq, tmp_str_seq);
}

template <typename Archive>
void Input1MLPModelNoFeature::save(Archive &ar, unsigned version) const 
{
    ar & word_embedding_dim &word_dict_size
        &input_dim &output_dim
        &mlp_hidden_dim_list
        &dropout_rate;
    ar &word_dict &postag_dict ;
    ar &context_feature;
    ar &*m;
}

template <typename Archive>
void Input1MLPModelNoFeature::load(Archive &ar, unsigned version)
{
    ar &word_embedding_dim &word_dict_size
        &input_dim &output_dim
        &mlp_hidden_dim_list
        &dropout_rate;
    ar &word_dict &postag_dict;
    assert(word_dict.size() == word_dict_size && postag_dict.size() == output_dim);
    ar &context_feature;
    build_model_structure();
    ar &*m;
}

template <typename Archive>
void Input1MLPModelNoFeature::serialize(Archive &ar, unsigned version)
{
    boost::serialization::split_member(ar, *this, version);
}

} // end of namespace slnn


#endif