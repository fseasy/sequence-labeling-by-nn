#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_FEATURE_H_

#include "lexicon_feature.h"
#include "modelmodule/context_feature.h"
#include "type_feature.h"

namespace slnn{

struct CWSFeatureDataSeq
{
    LexiconFeatureDataSeq lexicon_feature_data_seq;
    ContextFeatureDataSeq context_feature_data_seq;
    CharTypeFeatureDataSeq chartype_feature_data_seq;
    size_t size() const { return lexicon_feature_data_seq.size(); }
    const LexiconFeatureDataSeq& get_lexicon_feature_data_seq() const { return lexicon_feature_data_seq; };
    LexiconFeatureDataSeq& get_lexicon_feature_data_seq(){ return lexicon_feature_data_seq; }
    const ContextFeatureDataSeq& get_context_feature_data_seq() const { return context_feature_data_seq; }
    ContextFeatureDataSeq& get_context_feature_data_seq(){ return context_feature_data_seq; }
    const CharTypeFeatureDataSeq& get_chartype_feature_data_seq() const { return chartype_feature_data_seq; }
    CharTypeFeatureDataSeq& get_chartype_feature_data_seq(){ return chartype_feature_data_seq; }
};

class CWSFeatureLayer;

class CWSFeature
{
    friend class boost::serialization::access;
    friend class CWSFeatureLayer;
public:
    CWSFeature(DictWrapper &word_dict_wrapper);

    void set_feature_parameters(unsigned lexicon_start_here_dim, unsigned lexicon_pass_here_dim, unsigned lexicon_end_here_dim,
        unsigned context_left_size, unsigned context_right_size, unsigned word_embedding_dim,
        unsigned chartype_dim);
    unsigned get_feature_dim() const { return lexicon_feature.get_feature_dim() + context_feature.get_feature_dim()
                                              + chartype_feature.get_feature_dim(); };
    
    void count_word_frequency(const Seq &word_seq){ lexicon_feature.count_word_frequency(word_seq); };
    void build_lexicon(){ lexicon_feature.build_lexicon(); };
    void random_replace_with_unk(const CWSFeatureDataSeq &origin_cws_feature_seq, CWSFeatureDataSeq &replaced_cws_feature_seq);
    void extract(const Seq &char_seq, const IndexSeq &index_char_seq, CWSFeatureDataSeq &cws_feature_seq);
    
    std::string get_feature_info() const ;

    template <typename Archive>
    void serialize(Archive &ar, unsigned version);

    // DEBUG
    void debug_one_sent(const Seq &char_seq, const CWSFeatureDataSeq &cws_feature_seq)
    {
        lexicon_feature.debug_print_lexicon();
        lexicon_feature.debug_lexicon_feature_seq(char_seq, cws_feature_seq.get_lexicon_feature_data_seq());
        context_feature.debug_context_feature_seq(cws_feature_seq.get_context_feature_data_seq());
    }
private :
    LexiconFeature lexicon_feature;
    ContextFeature context_feature;
    CharTypeFeature chartype_feature;
};


inline
void CWSFeature::random_replace_with_unk(const CWSFeatureDataSeq &origin_cws_feature_seq, CWSFeatureDataSeq &replaced_cws_feature_seq)
{
    using std::swap;
    CWSFeatureDataSeq tmp_data_seq;
    context_feature.random_replace_with_unk(origin_cws_feature_seq.get_context_feature_data_seq(), 
        tmp_data_seq.get_context_feature_data_seq());
    assert(tmp_data_seq.get_context_feature_data_seq().size() > 0);
    tmp_data_seq.get_lexicon_feature_data_seq() = origin_cws_feature_seq.get_lexicon_feature_data_seq();
    tmp_data_seq.get_chartype_feature_data_seq() = origin_cws_feature_seq.get_chartype_feature_data_seq();
    swap(replaced_cws_feature_seq, tmp_data_seq);
}

template <typename Archive>
void CWSFeature::serialize(Archive &ar, unsigned version)
{
    ar & lexicon_feature & context_feature &chartype_feature;
}

} // end of namespace slnn


#endif