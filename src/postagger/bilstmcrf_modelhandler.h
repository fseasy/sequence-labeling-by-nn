#ifndef DOUBLECHANNEL_MODELHANDLER_H_INCLUDED_
#define DOUBLECHANNEL_MODELHANDLER_H_INCLUDED_

#include <boost/archive/text_oarchive.hpp>

#include "bilstmmodel4tagging_doublechannel.h"
#include "utils/utf8processing.hpp"
#include "utils/typedeclaration.h"

namespace slnn
{
struct DoubleChannelModelHandler
{
    DoubleChannelModel4POSTAG &dc_m;
    
    // Saving temporal model
    float best_acc;
    std::stringstream best_model_tmp_ss;

    // others 
    static const std::string number_transform_str;
    static const size_t length_transform_str;
    const size_t SentMaxLen = 256;
    const size_t MaxSentNum = 0x8FFFF;

    DoubleChannelModelHandler(DoubleChannelModel4POSTAG &dc_m);

    // Before read data
    void set_unk_replace_threshold(int freq_thres , float prob_thres);
    void build_fixed_dict_from_word2vec_file(std::ifstream &is);

    // Reading data 
    inline std::string replace_number(const std::string &str);
    void do_read_annotated_dataset(std::istream &is, std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
        std::vector<IndexSeq> &postag_seqs);
    void read_training_data_and_build_dynamic_and_postag_dicts(std::istream &is, std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
        std::vector<IndexSeq> &postag_seqs);
    void read_devel_data(std::istream &is, std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
        std::vector<IndexSeq> &postag_seqs);
    void read_test_data(std::istream &is, std::vector<Seq> &raw_test_sents, std::vector<IndexSeq> &daynamic_sents ,
        std::vector<IndexSeq> &fixed_sents);
    
    // After Reading Training data
    void finish_read_training_data(boost::program_options::variables_map &varmap);
    void build_model();
    void load_fixed_embedding(std::istream &is);

    // Train & devel & predict
    void train(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
        const std::vector<IndexSeq> *p_postag_seqs , 
        unsigned max_epoch, 
        const std::vector<IndexSeq> *p_dev_dynamic_sents=nullptr, const std::vector<IndexSeq> *p_dev_fixed_sents=nullptr,
        const std::vector<IndexSeq> *p_dev_postag_seqs=nullptr ,
        const unsigned long do_devel_freq = 50000);
    float devel(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
        const std::vector<IndexSeq> *p_postag_seqs,
        std::ostream *p_error_output_os = nullptr);
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);
private :
    inline void save_current_best_model(float acc);

};


inline
std::string DoubleChannelModelHandler::replace_number(const std::string &str)
{
    std::string tmp_str = str;
    size_t start_pos = 0;
    while (start_pos < tmp_str.length())
    {
        size_t end_pos = start_pos;

        while (true)
        {
            size_t byte_len = UTF8Processing::get_number_byte_width(tmp_str, end_pos);
            if (0 == byte_len) break;
            else end_pos += byte_len;
        }
        size_t number_byte_len = end_pos - start_pos;
        if (0 != number_byte_len)
        {
            // replace
            tmp_str.replace(start_pos, number_byte_len, number_transform_str);
            start_pos += length_transform_str;
        }
        else ++start_pos;
    }
    return tmp_str;
}


inline 
void DoubleChannelModelHandler::save_current_best_model(float acc)
{
    BOOST_LOG_TRIVIAL(info) << "better model has been found . stash it .";
    best_acc = acc;
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss);
    to << *dc_m.m;
}

} // end of namespace 
#endif
