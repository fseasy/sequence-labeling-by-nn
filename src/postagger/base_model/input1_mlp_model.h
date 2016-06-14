#ifndef POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_H_
#define POSTAGGER_BASEMODEL_INPUT1_MLP_MODEL_H_

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
    void set_replace_threshold(int freq_threshold, float prob_threshold);
    bool is_dict_frozen();
    void freeze_dict();
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0;

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;
    void input_seq2index_seq(const Seq &sent, const Seq &postag_seq, 
        IndexSeq &index_sent, IndexSeq &index_postag_seq, 
        ContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for annotated data

    void input_seq2index_seq(const Seq &sent, 
        IndexSeq &index_sent, 
        ContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq); // for input data

    void replace_word_with_unk(const IndexSeq &sent, 
        const ContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
        const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq,
        IndexSeq &replaced_sent,
        ContexFeature::ContextFeatureIndexGroupSeq &context_replaced_feature_gp_seq,
        POSFeature::POSFeatureIndexGroupSeq &replaced_feature_gp_seq);

    void postag_index_seq2postag_str_seq(const IndexSeq &postag_index_seq, Seq &postag_str_seq);


private :
    cnn::Dict word_dict;
    DictWrapper word_dict_wrapper;
};

}


#endif