#ifndef POS_MODEL_HANDLER_SINGLE_INPUT_WITH_FEATURE_MODELHANDLERS_HPP_
#define POS_MODEL_HANDLER_SINGLE_INPUT_WITH_FEATURE_MODELHANDLERS_HPP_
#include <fstream>
#include <vector>
#include "postagger/base_model/single_input_with_feature_model.hpp"

namespace slnn{

template <tepename RNNDerived, typename SIModel>
class SingleInputModelHandlerWithFeature
{
public :
    using POSFeatureExtractor::POSFeaturesIndexSeq;
    SingleInputModelWithFeature *sim;

    static const size_t MaxSentNum = 0x8000; // 32k
    static const std::string OUT_SPLIT_DELIMITER ;

    SingleINputModelHandlerWithFeature();
    ~SingleInputModelHandlerWithFeature();

    // Before read data
    void set_unk_replace_threshold(int freq_thres , float prob_thres);

    // Reading data 
    void read_annotated_data(std::ifstream &is,
                             std::vetor<IndexSeq> &sents,
                             std::vector<POSFeatureIndexSeq> &features_seqs,
                             std::vector<IndexSeq> &postags_seqs);
    void read_training_data(std::ifstream &is, 
                            std::vector<IndexSeq> &training_sents,
                            std::vector<POSFeaturesIndexSeq> &features_seqs,
                            std::vector<IndexSeq> &postags_seqs);
    void read_devel_data(std::ifstream &is,
                         std::vector<IndexSeq> &devel_sents,
                         std::vector<POSFeaturesIndexSeq> &features_seqs,
                         std::vector<IndexSeq> &postag_seqs);
    void read_test_data(std::ifstream &is,
                        std::vector<Seq> &raw_sents,
                        std::vector<IndexSeq> &sents,
                        std::vector<POSFeaturesIndexSeq> &features_seqs);

};


} // end of namespace slnn


#endif