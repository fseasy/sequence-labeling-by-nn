#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_

#include "lexicon_feature.h"

namespace slnn{
struct CWSFeatureDataSeq
{
    LexiconFeatureDataSeq lexicon_feature_data_seq;

    size_t size(){ return lexicon_feature_data_seq.size(); }
    const LexiconFeatureDataSeq& get_const_lexicon_feature_data_seq() const { return lexicon_feature_data_seq; }
    const LexiconFeatureDataSeq& get_lexicon_feature_data_seq() const { return get_const_lexicon_feature_data_seq(); };
    LexiconFeatureDataSeq& get_lexicon_feature_data_seq(){ return lexicon_feature_data_seq; }
};

class CWSFeatureLayer;

class CWSFeature
{
    friend class boost::serialization::access;
    friend class CWSFeatureLayer;
public:
    CWSFeature();

    void set_dim(unsigned start_here_dim, unsigned pass_here_dim, unsigned end_here_dim);
    unsigned get_feature_dim() const { return lexicon_feature.get_feature_dim(); };
    
    void count_word_freqency(const Seq &word_seq){ lexicon_feature.count_word_freqency(word_seq); };
    void build_lexicon(){ lexicon_feature.build_lexicon(); };

    void extract(const Seq &char_seq, CWSFeatureDataSeq &cws_feature_seq);
    
    std::string get_feature_info();

    template <typename Archive>
    void serialize(Archive &ar, unsigned version);

private :
    LexiconFeature lexicon_feature;
};


template <typename Archive>
void CWSFeature::serialize(Archive &ar, unsigned version)
{
    ar & lexicon_feature;
}

} // end of namespace slnn


#endif