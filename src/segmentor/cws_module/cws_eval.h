#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_EVAL_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_EVAL_H_
#include "token_module/cws_tag_definition.h"

namespace slnn{
namespace segmentor{
namespace eval{

namespace eval_inner{

struct EvalTempResultT
{
    unsigned nr_token_predict_right;
    unsigned nr_token_predict;
    unsigned nr_token_gold;
    unsigned nr_tag_predict_right;
    unsigned nr_tag;
    EvalTempResultT() :
        nr_token_predict_right(0), nr_token_predict(0), nr_token_gold(0), nr_tag_predict_right(0), nr_tag(0)
    {}
    void clear(){ nr_token_predict_right = nr_token_predict = nr_token_gold = nr_tag_predict_right = nr_tag = 0; }
    EvalTempResultT& operator+=(const EvalTempResultT &rhs)
    {
        this->nr_token_predict_right += rhs.nr_token_predict_right;
        this->nr_token_predict += rhs.nr_token_predict;
        this->nr_token_gold += rhs.nr_token_gold;
        this->nr_tag_predict_right += rhs.nr_tag_predict_right;
        this->nr_tag += rhs.nr_tag;
        return *this;
    }
};

} // end of namespace eval_inner

struct EvalResultT
{
    float p;
    float r;
    float f1;
    float acc;
    unsigned nr_token_predict_right;
    unsigned nr_token_predict;
    unsigned nr_token_gold;
    unsigned nr_tag_predict_right;
    unsigned nr_tag;
};

/**
 * do segmentor eval.
 * not thread safe.
 */
class SegmentorEval
{
public:
    SegmentorEval();
    SegmentorEval(const SegmentorEval &) = delete;
    SegmentorEval(const SegmentorEval &&) = delete;
    SegmentorEval& operator=(const SegmentorEval&) = delete;
public:
    // iteratively
    void start_eval();
    void eval_iteratively(const std::vector<Tag> &gold_tagseq, const std::vector<Tag> &pred_tagseq);
    EvalResultT end_eval();
    // batch
    EvalResultT eval(const std::vector<std::vector<Tag>> &gold_tagseq_set, const std::vector<std::vector<Tag>> &pred_tagseq);
private:
    eval_inner::EvalTempResultT eval_one(const std::vector<Tag> &gold_tagseq, const std::vector<Tag> &pred_tagseq);
    std::vector<std::pair<unsigned, unsigned>> tagseq2word_range_list(const std::vector<Tag> &seq);
private:
    eval_inner::EvalTempResultT tmp_result4iter;
};


/******************************************
 * Inline Implementation
 ******************************************/

inline
void SegmentorEval::start_eval()
{
    tmp_result4iter.clear();
}
} // end of namespace eval
} // end of namespace segmentor
} // end of namespace slnn


#endif