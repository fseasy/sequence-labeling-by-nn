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

using Str = std::u32string;
using StrSeq = std::vector<Str>;

struct UnannotatedInstance
{
    StrSeq word_seq;
    StrSeq pos_tag_seq;
    std::size_t size() const { return word_seq.size(); }
    void push_back(const Str &&word, const Str &&pos_tag)
    {
        word_seq.push_back(std::move(word)); pos_tag_seq.push_back(std::move(pos_tag));
    }
    Str to_string() const ;
};

struct AnnotatedInstance: public UnannotatedInstance
{
    StrSeq ner_tag_seq;
    void push_back(const Str &&word, const Str &&pos_tag, 
                   const Str &&ner_tag)
    {
        UnannotatedInstance::push_back(std::move(word), std::move(pos_tag));
        ner_tag_seq.push_back(std::move(ner_tag));
    }
    Str to_string() const ;
};

struct TokenDict
{
    friend boost::serialization::access;

    slnn::trivial::LookupTable<Str> word_dict;
    slnn::trivial::LookupTable<Str> pos_tag_dict;
    slnn::trivial::LookupTable<Str> ner_tag_dict;


    void freeze_and_set_unk()
    { 
        word_dict.freeze(); pos_tag_dict.freeze(); ner_tag_dict.freeze(); 
        word_dict.set_unk();
    }
    std::size_t word_num_with_unk() const { return word_dict.size(); }
    std::size_t pos_tag_num() const { return pos_tag_dict.size(); }
    std::size_t ner_tag_num() const { return ner_tag_dict.size(); }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int);
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

    std::size_t size() const { if( word_feat ){ return word_feat->size(); } else{ return 0U; } }
    Str to_string() const;
    Str to_char_string(const TokenDict&) const;

};

using NerTagIndex = int;
using NerTagIndexSeq = std::vector<NerTagIndex>;

struct WordFeatInfo
{
    std::vector<std::size_t> word_cnt_lookup;
    InstanceFeature::FeatIndex word_unk_index;
    std::size_t count(InstanceFeature::FeatIndex word_index) const { return word_cnt_lookup.at(word_index); }
    InstanceFeature::FeatIndex get_unk_index() const noexcept{ return word_unk_index; }
};

/*****
 * interface.
 ******/

AnnotatedInstance line2annotated_instance(const Str& uline);

UnannotatedInstance line2unannotated_instance(const Str& uline);

template<typename InstanceType>
std::vector<InstanceType>
dataset2instance_list(std::istream& is, InstanceType(*line2instance_func)(const Str&));

std::vector<AnnotatedInstance>
annotated_dataset2instance_list(std::ifstream &is);

std::vector<UnannotatedInstance>
unannotated_dataset2instance_list(std::ifstream &is);

std::shared_ptr<TokenDict> build_token_dict(const std::vector<AnnotatedInstance>&);

std::shared_ptr<WordFeatInfo> build_word_feat_info(const TokenDict&, const std::vector<InstanceFeature>&);

/**********
 *  feature, including word, pos-tag
 **********/
template<typename InstanceT>
InstanceFeature instance2feature(const InstanceT&, const TokenDict&);

NerTagIndexSeq ner_seq2ner_index_seq(const StrSeq&, const TokenDict&);
StrSeq ner_index_seq2ner_seq(const NerTagIndexSeq& ner_index_seq, const TokenDict&);


/***************************
 * Inline Implementation
 ***************************/

template <class Archive>
void TokenDict::serialize(Archive& ar, const unsigned int)
{
    ar &word_dict &pos_tag_dict &ner_tag_dict;
}


template<typename InstanceType>
std::vector<InstanceType>
dataset2instance_list(std::istream& is, 
    InstanceType(*line2instance_func)(const Str&))
{
    using std::swap;
    reader::NerUnicodeReader reader(is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(is));
    Str uline;
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
    std::size_t len = instance.size();
    feat.word_feat = std::make_shared<InstanceFeature::FeatSeq>(InstanceFeature::FeatSeq(len));
    feat.pos_tag_feat = std::make_shared<InstanceFeature::FeatSeq>(InstanceFeature::FeatSeq(len));
    for( std::size_t i = 0; i < len; ++i )
    {
        (*feat.word_feat)[i] = dict.word_dict.convert(instance.word_seq[i]);
        (*feat.pos_tag_feat)[i] = dict.pos_tag_dict.convert(instance.pos_tag_seq[i]);
    }
    return feat;
}

} // namespace token_module
} // namespace ner
} // namespace slnn


#endif