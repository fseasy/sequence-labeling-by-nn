#include "cws_eval.h"
namespace slnn{
namespace segmenter{
namespace eval{

SegmentorEval::SegmentorEval()
    :tmp_result4iter()
{}

void SegmentorEval::eval_iteratively(const std::vector<Index> &gold_tagseq, const std::vector<Index> &pred_tagseq)
{
    assert(gold_tagseq.size() == pred_tagseq.size());
    eval_inner::EvalTempResultT cur_tmp_result = eval_one(gold_tagseq, pred_tagseq);
    tmp_result4iter += cur_tmp_result;
}

EvalResultT SegmentorEval::end_eval()
{
    float Acc = (tmp_result4iter.nr_tag == 0) ? 0.f 
        : static_cast<float>(tmp_result4iter.nr_tag_predict_right) / tmp_result4iter.nr_tag * 100.f ;
    float P = (tmp_result4iter.nr_token_predict == 0) ? 0.f 
        : static_cast<float>(tmp_result4iter.nr_token_predict_right) / tmp_result4iter.nr_token_predict *100.f ;
    float R = (tmp_result4iter.nr_token_gold == 0) ? 0.f 
        : static_cast<float>(tmp_result4iter.nr_token_predict_right) / tmp_result4iter.nr_token_gold *100.f ;
    float F1 = (std::abs(R + P - 0.f) < 1e-6) ? 0.f : 2 * P * R / (P + R)  ;
    EvalResultT tmp;
    // assign
    tmp.p = P; tmp.r = R; tmp.f1 = F1; tmp.acc = Acc;
    tmp.nr_tag = tmp_result4iter.nr_tag; tmp.nr_tag_predict_right = tmp_result4iter.nr_tag_predict_right;
    tmp.nr_token_gold = tmp_result4iter.nr_token_gold; tmp.nr_token_predict = tmp_result4iter.nr_token_predict;
    tmp.nr_token_predict_right = tmp_result4iter.nr_token_predict_right;
    return tmp;
}
EvalResultT SegmentorEval::eval(const std::vector<std::vector<Index>> &gold_tagseq_set, const std::vector<std::vector<Index>> &pred_tagseq_set)
{
    eval_inner::EvalTempResultT backup = tmp_result4iter;
    start_eval();
    for( unsigned i = 0; i < gold_tagseq_set.size(); ++i )
    {
        eval_iteratively(gold_tagseq_set[i], pred_tagseq_set[i]);
    }
    EvalResultT result = end_eval();
    tmp_result4iter = backup;
    return result;
}

eval_inner::EvalTempResultT SegmentorEval::eval_one(const std::vector<Index> &gold_tagseq, const std::vector<Index> &pred_tagseq)
{
    assert(gold_tagseq.size() == pred_tagseq.size()) ;
    std::vector<std::pair<unsigned, unsigned>> gold_words = tagseq2word_range_list(gold_tagseq) ;
    std::vector<std::pair<unsigned, unsigned>> pred_words = tagseq2word_range_list(pred_tagseq) ;
    unsigned gold_word_size = gold_words.size(),
        pred_word_size = pred_words.size() ;
    size_t gold_pos = 0 ,
        pred_pos = 0 ;
    unsigned correct_cnt = 0 ;
    while( gold_pos < gold_word_size && pred_pos < pred_word_size )
    {
        std::pair<unsigned, unsigned> &gold_word = gold_words[gold_pos],
            &pred_word = pred_words[pred_pos] ;
        if( gold_word.first == pred_word.first )
        {
            // word is aligned
            if( gold_word.second == pred_word.second )
            {
                ++correct_cnt ;
            }
        }
        ++gold_pos ;
        if( gold_pos >= gold_word_size ) break ;
        // try to align
        unsigned gold_char_pos = gold_words[gold_pos].first ;
        while( pred_pos < pred_word_size && pred_words[pred_pos].first < gold_char_pos ) 
            ++pred_pos ;
    }
    eval_inner::EvalTempResultT tmp_result;
    tmp_result.nr_tag = pred_tagseq.size();
    for( unsigned i = 0; i < tmp_result.nr_tag; ++i ){ tmp_result.nr_tag_predict_right += (gold_tagseq[i] == pred_tagseq[i]); }
    tmp_result.nr_token_gold = gold_word_size;
    tmp_result.nr_token_predict = pred_word_size;
    tmp_result.nr_token_predict_right = correct_cnt;
    return tmp_result;
}

std::vector<std::pair<unsigned, unsigned>> SegmentorEval::tagseq2word_range_list(const std::vector<Index> &seq)
{
    std::vector<std::pair<unsigned, unsigned>> tmp_word_ranges ;
    unsigned range_s = 0 ;
    for( unsigned i = 0 ; i < seq.size() ; ++i )
    {
        Index tag_id = seq.at(i) ;
        if( tag_id == Tag::TAG_S_ID )
        {
            tmp_word_ranges.push_back({ i , i }) ;
            range_s = i + 1 ;
        }
        else if( tag_id == Tag::TAG_E_ID)
        {
            tmp_word_ranges.push_back({ range_s , i }) ;
            range_s = i + 1 ;
        }
    }
    return tmp_word_ranges;
}

}
} // end of nemspace segmenter
} // end of namespace slnn