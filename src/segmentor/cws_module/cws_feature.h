#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_

#include "lexicon_feature.h"
#include "modelmodule/context_feature.h"

namespace slnn{

struct CWSFeatureDataSeq
{
    LexiconFeatureDataSeq lexicon_feature_data_seq;
    ContextFeatureDataSeq context_feature_data_seq;
    size_t size(){ return lexicon_feature_data_seq.size(); }
    const LexiconFeatureDataSeq& get_lexicon_feature_data_seq() const { return lexicon_feature_data_seq; };
    LexiconFeatureDataSeq& get_lexicon_feature_data_seq(){ return lexicon_feature_data_seq; }
    const ContextFeatureDataSeq& get_context_feature_data_seq() const { return context_feature_data_seq; }
    ContextFeatureDataSeq& get_context_feature_data_seq(){ return context_feature_data_seq; }
};

class CWSFeatureLayer;

class CWSFeature
{
    friend class boost::serialization::access;
    friend class CWSFeatureLayer;
public:
    CWSFeature(DictWrapper &word_dict_wrapper);

    void set_feature_parameters(unsigned lexicon_start_here_dim, unsigned lexicon_pass_here_dim, unsigned lexicon_end_here_dim,
        unsigned context_left_size, unsigned context_right_size, unsigned word_embedding_dim);
    unsigned get_feature_dim() const { return lexicon_feature.get_feature_dim() + context_feature.get_context_feature_dim(); };
    
    void count_word_frequency(const Seq &word_seq){ lexicon_feature.count_word_frequency(word_seq); };
    void build_lexicon(){ lexicon_feature.build_lexicon(); };

    void extract(const Seq &char_seq, const IndexSeq &index_char_seq, CWSFeatureDataSeq &cws_feature_seq);
    
    std::string get_feature_info() const ;

    template <typename Archive>
    void serialize(Archive &ar, unsigned version);

    // DEBUG
    void debug_one_sent(const Seq &char_seq, const CWSFeatureDataSeq &cws_feature_seq)
    {
        lexicon_feature.debug_print_lexicon();
        lexicon_feature.debug_lexicon_feature_seq(char_seq, cws_feature_seq.lexicon_feature_data_seq);
    }
private :
    LexiconFeature lexicon_feature;
    ContextFeature context_feature;
};


template <typename Archive>
void CWSFeature::serialize(Archive &ar, unsigned version)
{
    ar & lexicon_feature & context_feature;
}

} // end of namespace slnn


#endif