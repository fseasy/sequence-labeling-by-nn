#include <vector>
#include <fstream>
#include <iostream>
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
    token_module::build_token_dict(instance_list);
    
    
    return 0;
}