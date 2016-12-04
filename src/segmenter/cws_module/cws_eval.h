#ifndef SLNN_SEGMENTER_CWS_MODULE_CWS_EVAL_H_
#define SLNN_SEGMENTER_CWS_MODULE_CWS_EVAL_H_
#include "token_module/cws_tag_definition.h"

namespace slnn{
namespace segmenter{
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
    float p = 0.f;
    float r = 0.f;
    float f1 = 0.f;
    float acc = 0.f;
    unsigned nr_token_predict_right = 0;
    unsigned nr_token_predict = 0;
    unsigned nr_token_gold = 0;
    unsigned nr_tag_predict_right = 0;
    unsigned nr_tag = 0;
};

/**
 * do segmenter eval.
 * not thread safe.
 */
class SegmenterEval
{
public:
    SegmenterEval();
    SegmenterEval(const SegmenterEval &) = delete;
    SegmenterEval(const SegmenterEval &&) = delete;
    SegmenterEval& operator=(const SegmenterEval&) = delete;
public:
    // iteratively
    void start_eval();
    void eval_iteratively(const std::vector<Index> &gold_tagseq, const std::vector<Index> &pred_tagseq);
    EvalResultT end_eval();
    // batch
    EvalResultT eval(const std::vector<std::vector<Index>> &gold_tagseq_set, const std::vector<std::vector<Index>> &pred_tagseq);
private:
    eval_inner::EvalTempResultT eval_one(const std::vector<Index> &gold_tagseq, const std::vector<Index> &pred_tagseq);
private:
    eval_inner::EvalTempResultT tmp_result4iter;
};


/******************************************
 * Inline Implementation
 ******************************************/

inline
void SegmenterEval::start_eval()
{
    tmp_result4iter.clear();
}


} // end of namespace eval
} // end of namespace segmenter
} // end of namespace slnn


#endif