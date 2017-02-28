#include <vector>
#include <fstream>
#include <iostream>
#include "ner/mlp/token_module.h"

namespace token_module = slnn::ner::token_module;
using slnn::ner::token_module::AnnotatedInstance;


int main()
{
    std::vector<AnnotatedInstance> instance_list;
    std::ifstream is("S:/GitHub/sequence-labeling-by-nn/run_flow_example/sampledata/ner/PKU_train.50.ner");
    if( !is )
    {
        throw std::runtime_error("failed to open ner sample data.");
    }
    token_module::read_annotated_data2raw_instance_list(is, instance_list);
    std::cout << "insance size " << instance_list.size() << std::endl;
    auto conv = slnn::charcode::CharcodeConvertor::create_convertor(slnn::charcode::EncodingType::UTF8);
    for( auto &instance : instance_list )
    {
        std::cout << conv->encode(instance.to_string()) << std::endl;
    }
    return 0;
}