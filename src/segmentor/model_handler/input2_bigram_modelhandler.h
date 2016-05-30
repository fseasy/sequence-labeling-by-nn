#ifndef SLNN_SEGMENTOR_MODEL_HANDLER_INPUT2_BIGRAM_MODELHANDLER_H
#define SLNN_SEGMENTOR_MODEL_HANDLER_INPUT2_BIGRAM_MODELHANDLER_H

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string/split.hpp>
#include "segmentor/model_handler/input2_modelhandler.h"

#include "utils/stat.hpp"
namespace slnn{

template <typename I2Model>
class Input2BigramModelHandler : public Input2ModelHandler<I2Model>
{

public :
    Input2BigramModelHandler() ;
    ~Input2BigramModelHandler() ;

    // Reading data 
    void do_read_annotated_dataset(std::istream &is, 
                                   std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
                                   std::vector<IndexSeq> &tag_seqs) override ;
    void read_test_data(std::istream &is, 
                        std::vector<Seq> &raw_test_sents, 
                        std::vector<IndexSeq> &daynamic_sents, std::vector<IndexSeq> &fixed_sents) override ;
};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{

template <typename I2Model>
Input2BigramModelHandler<I2Model>::Input2BigramModelHandler()
    : Input2ModelHandler()
{}

template <typename I2Model>
Input2BigramModelHandler<I2Model>::~Input2BigramModelHandler()
{}

template <typename I2Model>
void Input2BigramModelHandler<I2Model>::do_read_annotated_dataset(std::istream &is,
                                                            std::vector<IndexSeq> &dynamic_sents, 
                                                            std::vector<IndexSeq> &fixed_sents,
                                                            std::vector<IndexSeq> &tag_seqs)
{
    unsigned line_cnt = 0;
    std::string line;
    std::vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents,
        tmp_tag_seqs;
    IndexSeq dsent,
        fsent,
        tag_seq;
    // pre-allocation
    tmp_dynamic_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_fixed_sents.reserve(MaxSentNum);
    tmp_tag_seqs.reserve(MaxSentNum);

    dsent.reserve(SentMaxLen);
    fsent.reserve(SentMaxLen);
    tag_seq.reserve(SentMaxLen);

    DictWrapper &dynamic_dict_wrapper = i2m->get_dynamic_dict_wrapper() ;
    cnn::Dict &fixed_dict = i2m->get_fixed_dict();
    cnn::Dict &tag_dict = i2m->get_tag_dict();
    while (getline(is, line)) {
        if (0 == line.size()) continue;
        dsent.clear();
        fsent.clear();
        tag_seq.clear() ;
        std::istringstream iss(line) ;
        std::string words_line ;
        Seq tmp_word_cont,
            tmp_tag_cont ;
        while( iss >> words_line )
        {
            Seq char_cont_of_word,
                tag_cont_of_word;
            CWSTaggingSystem::parse_words2word_tag(words_line, char_cont_of_word, tag_cont_of_word) ;
            tmp_word_cont.insert(tmp_word_cont.end(), char_cont_of_word.begin(), char_cont_of_word.end());
            tmp_tag_cont.insert(tmp_tag_cont.end(), tag_cont_of_word.begin(), tag_cont_of_word.end());
        }
        // build bi-gram
        for( size_t i = 0 ; i < tmp_word_cont.size() - 1 ; ++i )
        {
            tmp_word_cont[i] += "_" + tmp_word_cont[i + 1];
        }
        tmp_word_cont[tmp_word_cont.size() - 1] += "_$";
        for( size_t i = 0 ; i < tmp_word_cont.size() ; ++i )
        {
            Index dword_id = dynamic_dict_wrapper.Convert(tmp_word_cont[i]) ;
            Index fword_id = fixed_dict.Convert(tmp_word_cont[i]);
            Index tag_id = tag_dict.Convert(tmp_tag_cont[i]) ;
            dsent.push_back(dword_id);
            fsent.push_back(fword_id);
            tag_seq.push_back(tag_id);
        }
        tmp_dynamic_sents.push_back(dsent);
        tmp_fixed_sents.push_back(fsent);
        tmp_tag_seqs.push_back(tag_seq);
        ++line_cnt;
        if(0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    std::swap(dynamic_sents, tmp_dynamic_sents);
    std::swap(fixed_sents, tmp_fixed_sents);
    std::swap(tag_seqs, tmp_tag_seqs);
}

template <typename I2Model>
void Input2BigramModelHandler<I2Model>::read_test_data(std::istream &is,
                                                 std::vector<Seq> &raw_test_sents,
                                                 std::vector<IndexSeq> &dsents,
                                                 std::vector<IndexSeq> &fsents)
{
    cnn::Dict &dword_dict = i2m->get_dynamic_dict() ;
    cnn::Dict &fword_dict = i2m->get_fixed_dict() ;
    std::string line ;
    std::vector<Seq> tmp_raw_sents ;
    std::vector<IndexSeq> tmp_dsents,
        tmp_fsents;

    Seq raw_sent ;
    IndexSeq dsent,
        fsent;
    while( getline(is, line) )
    {
        // do not skip empty line .
        CWSTaggingSystem::split_word(line, raw_sent) ;
        Seq raw_sent_bigram = raw_sent;
        for( size_t i = 0; i < raw_sent_bigram.size() - 1; ++i )
        {
            raw_sent_bigram[i] += "_" + raw_sent_bigram[i + 1];
        }
        raw_sent_bigram[raw_sent_bigram.size() - 1] += "_$";
        dsent.clear();
        fsent.clear();
        for( size_t i = 0 ; i < raw_sent_bigram.size() ; ++i )
        {
            dsent.push_back(dword_dict.Convert(raw_sent_bigram[i]));
            fsent.push_back(fword_dict.Convert(raw_sent_bigram[i]));
        }
        tmp_raw_sents.push_back(raw_sent) ;
        tmp_dsents.push_back(dsent);
        tmp_fsents.push_back(fsent);
    }
    std::swap(raw_test_sents, tmp_raw_sents) ;
    std::swap(dsents, tmp_dsents);
    std::swap(fsents, tmp_fsents);
}

} // end of namespace slnn
#endif
