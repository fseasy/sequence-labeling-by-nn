#include <vector>
#include <fstream>
#include <iostream>
#include <utils/general.hpp>
#include "ner/mlp/token_module.h"

namespace token_module = slnn::ner::token_module;
using slnn::ner::token_module::AnnotatedInstance;
using slnn::ner::token_module::UnannotatedInstance;

int main()
{
    // annoated

    std::vector<AnnotatedInstance> instance_list;
    std::ifstream is("S:/GitHub/sequence-labeling-by-nn/run_flow_example/sampledata/ner/PKU_train.50.ner");
    if( !is )
    {
        throw std::runtime_error("failed to open ner sample data.");
    }
    instance_list = token_module::annotated_dataset2instance_list(is);
    std::cout << "insance size " << instance_list.size() << std::endl;
    auto conv = slnn::charcode::CharcodeConvertor::create_convertor(slnn::charcode::EncodingType::UTF8);
    for( auto &instance : instance_list )
    {
        std::cout << conv->encode(instance.to_string()) << std::endl;
    }
    is.close();

    std::cout << " +++++++++  unannotated +++++++++ " << std::endl;

    // unannotated
    std::vector<UnannotatedInstance> u_instance_list;
    std::ifstream input_is("S:/GitHub/sequence-labeling-by-nn/run_flow_example/sampledata/ner/PKU_input.50.ner");
    if( ! input_is )
    {
        throw std::runtime_error("failed to open ner sample data.");
    }
    u_instance_list = token_module::unannotated_dataset2instance_list(input_is);
    std::cout << "insance size " << instance_list.size() << std::endl;
    for( auto &instance : u_instance_list )
    {
        std::cout << conv->encode(instance.to_string()) << std::endl;
    }
    input_is.close();
    
    // token dict
    auto token_dict = token_module::build_token_dict(instance_list);
    
    // Instance 2 feature
    std::cout << std::endl;
    auto feat = token_module::instance2feature(instance_list[0], token_dict);
    std::cout << conv->encode(feat.to_string()) << std::endl;
    std::cout << conv->encode(feat.to_char_string(token_dict)) << std::endl;

    feat = token_module::instance2feature(u_instance_list[1], token_dict);
    std::cout << conv->encode(feat.to_string()) << std::endl;
    std::cout << conv->encode(feat.to_char_string(token_dict)) << std::endl;

    // get ner gold index sequence
    std::cout << std::endl;
    auto index_seq = token_module::ner_seq2ner_index_seq(instance_list[0].ner_tag_seq, token_dict);
    auto ner_tag_seq = token_module::ner_index_seq2ner_seq(index_seq, token_dict);
    std::cout << conv->encode(slnn::join(instance_list[0].ner_tag_seq, U' ')) << std::endl;
    std::cout << conv->encode(slnn::join(ner_tag_seq, U' ')) << std::endl;


    return 0;
}