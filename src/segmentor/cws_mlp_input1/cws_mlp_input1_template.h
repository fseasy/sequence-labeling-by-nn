#ifndef SLNN_SEGMENTOR_CWS_MLP_INPUT1_TEMPLATE_H_
#define SLNN_SEGMENTOR_CWS_MLP_INPUT1_TEMPLATE_H_
#include <random>
#include <boost/program_options/variables_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "segmentor/cws_module/token_module/cws_tag_definition.h"
namespace slnn{
namespace segmentor{
namespace mlp_input1{

template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
class SegmentorMlpInput1Template
{
public:
    // typename
    using AnnotatedDataProcessedT = typename TokenModuleT::AnnotatedDataProcessedT;
    using AnnotatedDataRawT = typename TokenModuleT::AnnotatedDataRawT;
    using UnannotatedDataProcessedT = typename TokenModuleT::UnannotatedDataProcessedT;
    using UnannotatedDataRawT = typename TokenModuleT::UnannotatedDataRawT;
public:
    TokenModuleT* get_token_module() const { return &token_module; }
    NnModuleT* get_nn() const { return &nn; }
    StructureParamT *get_param() const { return &param; }
    std::mt19937* get_mt19937_rng() const { static rng = std::mt19937(rng_seed); return &rng; }
    unsigned get_rng_seed() const { return rng_seed; }
private:
    explicit SegmentorMlpInput1Template(int argc, char **argv, unsigned seed) ;
public:
    SegmentorMlpInput1Template(const SegmentorMlpInput1Template&) = delete;
    SegmentorMlpInput1Template(SegmentorMlpInput1Template &&) = delete;
    SegmentorMlpInput1Template& operator=(const SegmentorMlpInput1Template&) = delete;
public:
    static std::shared_ptr<SegmentorMlpInput1Template> create_new_model(int argc, char**argv, unsigned seed);
    static void save_model(std::ostream &os, const SegmentorMlpInput1Template &m);
    static std::shared_ptr<SegmentorMlpInput1Template> load_and_build_model(std::istream &is, int argc, char **argv);
    void save_model(std::ostream &os) const { save_model(os, *this); }
public:
    void set_model_structure_param_from_outer(const boost::program_options::variables_map &args);
    void finish_read_training_data();
    void build_model_structure();
    void build_training_graph(const AnnotatedDataProcessedT& ann_processed_data);
    std::vector<Tag> predict(const UnannotatedDataProcessedT& unann_processed_data);
private:
    TokenModuleT token_module;
    StructureParamT param;
    NnModuleT nn;
    unsigned  rng_seed;
};

/**************************************************
 * Inline Implemnentation
 **************************************************/

template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
SegmentorMlpInput1Template(int argc, char **argv, unsigned seed)
    :token_module(seed),
    nn(argc, argv, seed),
    rng_seed(seed)
{}


template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
std::shared_ptr<SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>>
SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
create_new_model(int argc, char **argv, unsigned seed)
{
    return std::shared_ptr<SegmentorMlpInput1Template>(new SegmentorMlpInput1Template(argc, argv, seed));
}


template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
void 
SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
save_model(std::ostream &os, const SegmentorMlpInput1Template &m)
{
    boost::archive::text_oarchive to(os);
    // 1. first save the seed.
    to << m.get_rng_seed();
    // 2. save param
    to << *m.get_param();
    // 3. save token module
    to << *m.get_token_module();
    // 4. save nn
    to << *m.get_nn();
}

template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
std::shared_ptr<SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>> 
SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
load_and_build_model(std::istream &is, int argc, char **argv)
{
    boost::archive::text_iarchive ti(is);
    // 1. read seed
    unsigned seed;
    ti >> seed;
    // 2. create model by the seed
    std::shared_ptr<SegmentorMlpInput1Template> m = create_new_model(argc, argv, seed);
    // 3. read structure param
    ti >> *m->get_param();
    // 4. read token module
    ti >> *m->get_token_module();
    // 5. build structure
    m->build_model_structure();
    // 6. read nn
    ti >> *m->get_nn();
    return m;
}


template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
inline
void SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
set_model_structure_param_from_outer(const boost::program_options::variables_map &args)
{
    param.set_param_from_user_defined(args);
}


template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
inline
void SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
finish_read_training_data()
{
    param.set_param_from_token_module(token_module);
    token_module.finish_read_training_data();
}


template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
inline
void SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
build_model_structure()
{
    /*****
     * May be we should define 2 function for training and predicting progress.
     *****/
    token_module.set_unk_replace_threshold(param);
    nn.build_model_structure(param);
    std::cerr << param.get_structure_info() << "\n";
}

template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
inline
void SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
build_training_graph(const AnnotatedDataProcessedT& ann_processed_data)
{
    typename AnnotatedDataProcessedT data_after_unk_replace = 
        token_module.replace_low_freq_token2unk(ann_processed_data);
    nn.build_training_graph(data_after_unk_replace);
}

template <typename TokenModuleT, typename StructureParamT, typename NnModuleT>
inline
std::vector<Tag> SegmentorMlpInput1Template<TokenModuleT, StructureParamT, NnModuleT>::
predict(const UnannotatedDataProcessedT & unann_processed_data)
{
    return nn.predict(unann_processed_data);
}


} // enf of namespace mlp-input1
} // end of namespace segmentor
} // end of namespace slnn


#endif