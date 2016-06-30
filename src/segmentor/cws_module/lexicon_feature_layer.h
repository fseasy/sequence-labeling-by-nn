#ifndef SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_LAYER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_LAYER_H_
#include "lexicon_feature.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
namespace slnn{

class LexiconFeatureLayer
{
public :
    LexiconFeatureLayer(cnn::Model *cnn_m, unsigned start_here_dict_size, unsigned start_here_dim,
        unsigned pass_here_dict_size, unsigned pass_here_dim,
        unsigned end_here_dict_size, unsigned end_here_dim);
    LexiconFeatureLayer(cnn::Model *cnn_m, const LexiconFeature &lexicon_feature);
    void new_graph(cnn::ComputationGraph &cg);
    void build_lexicon_feature(const LexiconFeatureDataSeq &lexicon_feature_seq,
        std::vector<cnn::expr::Expression> &lexicon_feature_exprs);

private:
    cnn::LookupParameters *start_here_lookup_param;
    cnn::LookupParameters *pass_here_lookup_param;
    cnn::LookupParameters *end_here_lookup_param;
    cnn::ComputationGraph *pcg;
};

inline
void LexiconFeatureLayer::new_graph(cnn::ComputationGraph &cg)
{
    pcg = &cg;
}

inline
void LexiconFeatureLayer::build_lexicon_feature(const LexiconFeatureDataSeq &lexicon_feature_seq,
    std::vector<cnn::expr::Expression> &lexicon_feature_exprs)
{
    using std::swap;
    size_t seq_len = lexicon_feature_seq.size();
    std::vector<cnn::expr::Expression> tmp_lexicon_feature_exprs(seq_len);
    for( size_t i = 0; i < seq_len; ++i )
    {
        const LexiconFeatureData &fdata = lexicon_feature_seq.at(i);
        tmp_lexicon_feature_exprs[i] = cnn::expr::concatenate({
            cnn::expr::lookup(*pcg, start_here_lookup_param, fdata.get_start_here_feature_index()),
            cnn::expr::lookup(*pcg, pass_here_lookup_param, fdata.get_pass_here_feature_index()),
            cnn::expr::lookup(*pcg, end_here_lookup_param, fdata.get_end_here_feature_index())
        });
    }
    swap(lexicon_feature_exprs, tmp_lexicon_feature_exprs);
}

} // end of namespace slnn

#endif