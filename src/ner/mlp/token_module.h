#ifndef SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_
#define SLNN_NER_MLP_TOKEN_MODULE_INCLUDE_

#include <vector>
#include <string>
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
 * WORD_POS#NER
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



} // namespace token_module
} // namespace ner
} // namespace slnn


#endif