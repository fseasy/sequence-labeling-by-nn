#ifndef SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_
#define SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include "ner/module/ner_reader.h"
#include "trivial/lookup_table/lookup_table.h"

namespace slnn{
namespace ner{
namespace token_module{

/**
 * Token Module
 * 1. Storing the raw instance -> annotated instance, unannotated instance 
 * 2. build the dict -> word_dict, postag_dict, ner_tag_dict.
 * 3. build the index instance -> anntated instance, unannotated instance.
 **/


constexpr char32_t PIECE_DELIM = U'\t';
constexpr char32_t TRAIN_WORD_POS_DELIM = U'/';
constexpr char32_t TEST_WORD_POS_DELIM = U'_';
constexpr char32_t POS_NER_DELIM = U'#';


/**
* uannotated instance
* annotated instance structure. no copy, no share. 
* WORD_POS#NER\tWORD_POS#NER
*
**/
struct UnannotatedInstance
{
    std::vector<std::u32string> word_seq;
    std::vector<std::u32string> pos_tag_seq;
    std::size_t size(){ return word_seq.size(); }
    void push_back(const std::u32string &&word, const std::u32string &&pos_tag)
    {
        word_seq.push_back(std::move(word)); pos_tag_seq.push_back(std::move(pos_tag));
    }
    std::u32string to_string();
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
    std::u32string to_string();
};

struct TokenDict
{
    slnn::trivial::LookupTable<std::u32string> word_dict;
    slnn::trivial::LookupTable<std::u32string> pos_tag_dict;
    slnn::trivial::LookupTable<std::u32string> ner_tag_dict;


    void freeze_and_set_unk()
    { 
        word_dict.freeze(); pos_tag_dict.freeze(); ner_tag_dict.freeze(); 
        word_dict.set_unk();
    }
    std::size_t word_num_with_unk(){ return word_dict.size(); }
    std::size_t pos_tag_num(){ return pos_tag_dict.size(); }
    std::size_t ner_tag_num(){ return ner_tag_dict.size(); }
};

/******
 * will-copy, change partially( replace low-freq feature to )
 ************/
struct InstanceFeature
{
    using FeatIndex = int;
    using FeatSeq = std::vector<FeatIndex>;
    std::shared_ptr<FeatSeq> word_feat;
    std::shared_ptr<FeatSeq> pos_tag_feat;
    std::shared_ptr<FeatSeq> ner_tag_feat;
};

/*****
 * interface.
 ******/

AnnotatedInstance line2annotated_instance(const std::u32string& uline);

UnannotatedInstance line2unannotated_instance(const std::u32string &uline);

template<typename InstanceType>
std::vector<InstanceType>
dataset2instance_list(std::istream& is, InstanceType(*line2instance_func)(const std::u32string&));

std::vector<AnnotatedInstance>
annotated_dataset2instance_list(std::ifstream &is);

std::vector<UnannotatedInstance>
unannotated_dataset2instance_list(std::ifstream &is);

TokenDict build_token_dict(const std::vector<AnnotatedInstance>&);

template<typename InstanceT>
InstanceFeature instance2feature(const InstanceT&, const TokenDict&);

/***************************
 * Inline Implementation
 ***************************/

template<typename InstanceType>
std::vector<InstanceType>
dataset2instance_list(std::istream& is, 
    InstanceType(*line2instance_func)(const std::u32string&))
{
    using std::swap;
    reader::NerUnicodeReader reader(is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(is));
    std::u32string uline;
    std::size_t line_cnt = 0;
    std::vector<InstanceType> instance_list;

    while( reader.readline(uline) )
    {
        ++line_cnt;
        std::size_t len = uline.length();
        if( len == 0 ){ continue; }
        // Annotated Instance has implicitly-defined Move-Assignment Operator
        try
        {
            instance_list.push_back(line2instance_func(uline));
        }
        catch( std::runtime_error & e )
        {
            // catch and add the line cnt info.
            std::ostringstream oss;
            oss << e.what() << " at line: " << line_cnt;
            throw std::runtime_error(oss.str());
        }

    }
    return instance_list;
}

inline
std::vector<AnnotatedInstance>
annotated_dataset2instance_list(std::ifstream &is)
{
    return dataset2instance_list(is, line2annotated_instance);
}

inline 
std::vector<UnannotatedInstance>
unannotated_dataset2instance_list(std::ifstream &is)
{
    return dataset2instance_list(is, &line2unannotated_instance);
}


template<typename InstanceT>
InstanceFeature instance2feature(const InstanceT& instance, const TokenDict& dict)
{
    InstanceFeature feat;

}

} // namespace token_module
} // namespace ner
} // namespace slnn


#endif