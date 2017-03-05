#ifndef SLNN_NER_MODLE_UTILITY_INCLUDE_
#define SLNN_NER_MODLE_UTILITY_INCLUDE_

#include "token_module.h"
#include "structure_param_module.h"
#include "nn_module.h"

namespace slnn{
namespace ner{
namespace utility{

using token_module::AnnotatedInstance; 
using token_module::UnannotatedInstance;
using token_module::InstanceFeature;
using token_module::NerTagIndex;
using token_module::NerTagIndexSeq;
using token_module::TokenDict;
using token_module::WordFeatInfo;

using token_module::annotated_dataset2instance_list;
using token_module::unannotated_dataset2instance_list;
using token_module::instance2feature;
using token_module::build_token_dict;
using token_module::build_word_feat_info;

void annotated_instance_list2feat_list_and_ner_tag_list(
    const std::vector<AnnotatedInstance>&, const TokenDict&,
    std::vector<InstanceFeature>&, std::vector<NerTagIndexSeq>&);
void unannotated_instance_list2feat_list(
    const std::vector<UnannotatedInstance>&, const TokenDict&,
    std::vector<InstanceFeature>&);

using structure_param::StructureParam;
using structure_param::set_param_from_cmdline;
using structure_param::set_param_from_token_dict;

using nn::NnMlp;

void save(std::ostream&, const TokenDict&, const StructureParam&,
    NnMlp&);
void load(std::istream&, std::shared_ptr<TokenDict>&, std::shared_ptr<StructureParam>&,
    std::shared_ptr<NnMlp>&);



} // end of namespace utility
} // end of namespace ner
} // end of namespace slnn



#endif