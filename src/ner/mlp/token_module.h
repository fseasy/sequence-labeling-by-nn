#ifndef SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_
#define SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_

#include <vector>
#include <string>
#include <iostream>

#include "ner/module/ner_reader.h"

namespace slnn{
namespace ner{
namespace token_module{

/**
 * Token Module
 * 1. Storing the raw instance -> annotated instance, unannotated instance 
 * 2. build the dict -> word_dict, postag_dict, ner_tag_dict.
 * 3. build the index instance -> anntated instance, unannotated instance.
 **/

/**
 * annotated instance structure. no copy, no share. 
 * WORD_POS#NER\tWORD_POS#NER
 *
 **/

const char32_t PIECE_DELIM = U'\t';
const char32_t WORD_POS_DELIM = U'_';
const char32_t POS_NER_DELIM = U'#';


struct UnannotatedInstance
{
    std::vector<std::u32string> word_seq;
    std::vector<std::u32string> pos_tag_seq;
    std::size_t size(){ return word_seq.size(); }
    void push_back(const std::u32string &&word, const std::u32string &&pos_tag)
    {
        word_seq.push_back(std::move(word)); pos_tag_seq.push_back(std::move(pos_tag));
    }
};

struct AnnotatedInstance: public UnannotatedInstance
{
    std::vector<std::u32string> ner_tag_seq;
    void push_back(const std::u32string &&word, const std::u32string &&pos_tag, 
                   const std::u32string &&ner_tag)
    {
        UnannotatedInstance::push_back(std::move(word), std::move(pos_tag));
        ner_tag_seq.push_back(std::move(ner_tag));
    }
};

void read_annotated_data2raw_instance_list(std::istream &is,
    std::vector<AnnotatedInstance> &raw_instance_list)
{
    using std::swap;
    reader::NerUnicodeReader reader(is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(is));
    std::u32string uline;
    std::size_t line_cnt = 0;
    std::vector<AnnotatedInstance> instance_list;
    auto get_token = [](const std::u32string &piece, std::u32string& word,
        std::u32string& pos_tag, std::u32string& ner_tag)
    {
        //TODO: FINISH it !
    };
    while( reader.readline(uline) )
    {
        ++line_cnt;
        std::size_t len = uline.length();
        if( len == 0 ){ continue; }
        AnnotatedInstance instance;
        // WORD_POS#NER\tWORD_POS#NER
        std::size_t piece_spos = 0U,
            piece_epos = uline.find(piece_spos, PIECE_DELIM);
        while( piece_epos != std::u32string::npos )
        {
            auto underline_pos = uline.rfind(WORD_POS_DELIM, piece_epos);
            auto sharp_pos = uline.rfind(POS_NER_DELIM, piece_epos);
            if( underline_pos == std::u32string::npos ||
                underline_pos <= piece_spos ||
                sharp_pos == std::u32string::npos ||
                sharp_pos <= underline_pos )
            {
                // no toleration for training data.
                throw std::runtime_error("ill-formated annotated instance at line: " +
                    std::to_string(line_cnt));
            }
            instance.push_back(uline.substr(piece_spos, underline_pos - piece_spos),
                uline.substr(underline_pos + 1, sharp_pos - underline_pos - 1),
                uline.substr(sharp_pos + 1, piece_epos - sharp_pos - 1));
            piece_spos = piece_epos + 1;
            piece_epos = uline.find(piece_spos, PIECE_DELIM);
        }
        // the last part
        auto underline_pos = uline.rfind(WORD_POS_DELIM);
        auto sharp_pos = uline.rfind()
        std::size_t underline_pos = uline.rfind(U"_");
        std::size_t sharp_pos = u
    }
}

} // namespace token_module
} // namespace ner
} // namespace slnn


#endif