#ifndef SLNN_SEGMENTOR_CWS_MLP_INPUT1_TEMPLATE_H_
#define SLNN_SEGMENTOR_CWS_MLP_INPUT1_TEMPLATE_H_
#include <random>
#include <boost/program_options/variables_map.hpp>
#include "segmentor/cws_module/token_module/cws_tag_definition.h"
namespace slnn{
namespace segmentor{
namespace mlp_input1{
template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
class SegmentorMlpInput1Template
{
public:
    // typename
    using AnnotatedDataProcessedT = TokenModuleT::AnnotatedDataProcessedT;
    using AnnotatedDataRawT = TokenModuleT::AnnotatedDataRawT;
    using UnannotatedDataProcessedT = TokenModuleT::UnannotatedDataProcessedT;
    using UnannotatedDataRawT = TokenModuleT::UnannotatedDataRawT;

public:
    TokenModuleT* get_token_module(){ return &token_module; }
    NnModuleT* get_nn(){ return &nn; }
    StructureParamT *get_param(){ return &param; }
    std::mt19937& get_mt19937_rng(){ return &rng; }
public:
    SegmentorMlpInput1Template(unsigned seed);
    SegmentorMlpInput1Template(const SegmentorMlpInput1Template&) = delete;
    SegmentorMlpInput1Template(SegmentorMlpInput1Template &&) = delete;
    SegmentorMlpInput1Template& operator=(const SegmentorMlpInput1Template&) = delete;
public:
    void set_model_structure_param_from_outer(const boost::program_option::variable_map &args);
    void finish_read_training_data();
    void build_model_structure();
    void build_training_graph(const AnnotatedDataProcessedT& ann_processed_data);
    std::vector<Tag> predict(const UnannotatedDataProcessedT & unann_processed_data);
private:
    TokenModuleT token_module;
    StructureParamT param;
    NnModuleT nn;
    std::mt19937 rng;
};

} // enf of namespace mlp-input1
} // end of namespace segmentor
} // end of namespace slnn


#endif