#ifndef DOUBLECHANNEL_MODELHANDLER_H_INCLUDED_
#define DOUBLECHANNEL_MODELHANDLER_H_INCLUDED_

#include <string>
#include <vector>

#include "bilstmmodel4tagging_doublechannel.h"
#include "utils/utf8processing.hpp"
#include "utils/typedeclaration.h"

using namespace std;

namespace slnn
{
struct DoubleChannelModelHandler
{
    DoubleChannelModel4POSTAG &dc_m;
    
    // Saving temporal model
    float best_acc;
    stringstream best_model_tmp_ss;

    // others 
    static const string number_transform_str;
    static const size_t length_transform_str;

    DoubleChannelModelHandler(DoubleChannelModel4POSTAG &dc_m);

    // Reading data 
    inline string replace_number(const string &str);
    void do_read_annotated_dataset(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
        vector<IndexSeq> &postag_seqs);
    void read_training_data_and_build_dicts(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
        vector<IndexSeq> &postag_seqs);
    void read_devel_data(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
        vector<IndexSeq> &postag_seqs);
    void read_test_data(istream &is, vector<Seq> &raw_test_sents, vector<IndexSeq> &daynamic_sents ,
        vector<IndexSeq> &fixed_sents);
    
    // After Reading Training data
    void finish_read_training_data();

    // Train & devel & predict
    void train(const vector<InstancePair> *p_samples, unsigned max_epoch, const vector<InstancePair> *p_dev_samples = nullptr,
        const unsigned long do_devel_freq = 50000);
    float devel(const vector<InstancePair> *dev_samples, ostream *p_error_output_os = nullptr);
    void predict(istream &is, ostream &os);

};

const string DoubleChannelModelHandler::number_transform_str = "##";
const size_t DoubleChannelModelHandler::length_transform_str = number_transform_str.length();

inline
string DoubleChannelModelHandler::replace_number(const string &str)
{
    string tmp_str = str;
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

} // end of namespace 
#endif
