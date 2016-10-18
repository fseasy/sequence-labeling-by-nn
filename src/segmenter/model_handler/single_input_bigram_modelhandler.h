#ifndef SLNN_SEGMENTOR_CWS_SINGLE_BIGRAM_CLASSIFICATION_SINGLE_INPUT_BIGRAM_MODELHANDLER_H
#define SLNN_SEGMENTOR_CWS_SINGLE_BIGRAM_CLASSIFICATION_SINGLE_INPUT_BIGRAM_MODELHANDLER_H

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "segmenter/model_handler/single_input_modelhandler.h"

namespace slnn{

template <typename SIModel>
class SingleInputBigramModelHandler : public SingleInputModelHandler<SIModel>
{
public :
    SingleInputBigramModelHandler() ;
    ~SingleInputBigramModelHandler() ;

    // Reading data 
    void do_read_annotated_dataset(std::istream &is, 
                                   std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs) override ;

    void read_test_data(std::istream &is,
                        std::vector<Seq> &raw_test_sents, std::vector<IndexSeq> &sents) override;

};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{

template <typename SIModel>
SingleInputBigramModelHandler<SIModel>::SingleInputBigramModelHandler()
    : SingleInputModelHandler<SIModel>()
{}

template <typename SIModel>
SingleInputBigramModelHandler<SIModel>::~SingleInputBigramModelHandler(){}

template <typename SIModel>
void SingleInputBigramModelHandler<SIModel>::do_read_annotated_dataset(std::istream &is,
                    std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs) 
{
    unsigned line_cnt = 0;
    std::string line;
    std::vector<IndexSeq> tmp_sents,
        tmp_tag_seqs;
    IndexSeq sent, 
        tag_seq;
    // pre-allocation
    tmp_sents.reserve(SingleInputModelHandler<SIModel>::MaxSentNum); // 2^19 =  480k pairs 
    tmp_tag_seqs.reserve(SingleInputModelHandler<SIModel>::MaxSentNum);

    sent.reserve(SingleInputModelHandler<SIModel>::SentMaxLen);
    tag_seq.reserve(SingleInputModelHandler<SIModel>::SentMaxLen);

    DictWrapper &word_dict_wrapper = SingleInputModelHandler<SIModel>::sim->get_input_dict_wrapper() ;
    dynet::Dict &tag_dict = SingleInputModelHandler<SIModel>::sim->get_output_dict() ;
    while (getline(is, line)) {
        if (0 == line.size()) continue;
        sent.clear() ;
        tag_seq.clear() ;
        std::istringstream iss(line) ;
        std::string words_line ;
        Seq tmp_word_cont,
            tmp_tag_cont ;
        while( iss >> words_line )
        {
            Seq seg_word_cont,
                seg_tag_cont ;
            CWSTaggingSystem::parse_words2word_tag(words_line, seg_word_cont, seg_tag_cont) ;
            tmp_word_cont.insert(tmp_word_cont.end(), seg_word_cont.begin(), seg_word_cont.end());
            tmp_tag_cont.insert(tmp_tag_cont.end(), seg_tag_cont.begin(), seg_tag_cont.end());
        }
        // build bigram
        for( size_t i = 0; i < tmp_word_cont.size() - 1; ++i )
        {
            tmp_word_cont[i] += "_" + tmp_word_cont[i + 1];
        }
        tmp_word_cont[tmp_word_cont.size() - 1] += "_$";
        for( size_t i = 0 ; i < tmp_word_cont.size() ; ++i )
        {
            Index word_id = word_dict_wrapper.Convert(tmp_word_cont[i]) ;
            Index tag_id = tag_dict.Convert(tmp_tag_cont[i]) ;
            sent.push_back(word_id) ;
            tag_seq.push_back(tag_id) ;
        }
        tmp_sents.push_back(sent);
        tmp_tag_seqs.push_back(tag_seq);
        ++line_cnt;
        if(0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    std::swap(sents, tmp_sents);
    std::swap(tag_seqs, tmp_tag_seqs);
}

template <typename SIModel>
void SingleInputBigramModelHandler<SIModel>::read_test_data(std::istream &is,
                                                      std::vector<Seq> &raw_test_sents, 
                                                      std::vector<IndexSeq> &sents)
{
    dynet::Dict &word_dict = SingleInputModelHandler<SIModel>::sim->get_input_dict() ;
    std::string line ;
    std::vector<Seq> tmp_raw_sents ;
    std::vector<IndexSeq> tmp_sents ;

    Seq raw_sent ;
    IndexSeq sent ;
    while( getline(is, line) )
    {
        // do not skip empty line .
        CWSTaggingSystem::split_word(line, raw_sent) ;
        Seq raw_sent_bigram = raw_sent;
        // build bigram
        for( size_t i = 0 ; i < raw_sent_bigram.size()-1; ++i )
        {
            raw_sent_bigram[i] += "_" + raw_sent_bigram[i + 1];
        }
        raw_sent_bigram[raw_sent_bigram.size() - 1] += "_$";
        sent.clear() ;
        for( size_t i = 0 ; i < raw_sent_bigram.size() ; ++i )
        {
            sent.push_back(word_dict.Convert(raw_sent_bigram[i])) ;
        }
        tmp_raw_sents.push_back(raw_sent) ; // still using raw sent for output 
        tmp_sents.push_back(sent) ;
    }
    std::swap(raw_test_sents, tmp_raw_sents) ;
    std::swap(sents, tmp_sents) ;
}

} // end of namespace slnn
#endif
