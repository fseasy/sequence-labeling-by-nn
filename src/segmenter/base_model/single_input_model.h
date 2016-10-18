#ifndef SLNN_SEGMENTOR_BASEMODEL_SINGLE_INPUT_MODEL_H_
#define SLNN_SEGMENTOR_BASEMODEL_SINGLE_INPUT_MODEL_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"
#include "segmenter/cws_module/cws_tagging_system.h"
namespace slnn{

class SingleInputModel
{
    friend class boost::serialization::access;
public :
    SingleInputModel() ;
    virtual ~SingleInputModel() ;
    SingleInputModel(const SingleInputModel&) = delete;
    SingleInputModel& operator=(const SingleInputModel&) = delete;
    
    virtual void set_model_param(const boost::program_options::variables_map &var_map);

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;
    
    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg,
                                              const IndexSeq &input_seq, const IndexSeq &gold_seq) ;
    virtual void predict(dynet::ComputationGraph &cg ,
                         const IndexSeq &input_seq, IndexSeq &pred_seq) ;

    dynet::Dict& get_input_dict(){ return input_dict ;  } 
    dynet::Dict& get_output_dict(){ return output_dict ; } 
    DictWrapper& get_input_dict_wrapper(){ return input_dict_wrapper ; } 
    dynet::Model *get_dynet_model(){ return m ; } ;
    CWSTaggingSystem& get_tag_sys(){ return tag_sys ; }

    void set_dynet_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } 

    template <typename Archive>
    void save(Archive &ar, const unsigned versoin) const; 
    template <typename Archive>
    void load(Archive &ar, const unsigned version);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public :
    static const std::string UNK_STR;

public:
    unsigned word_embedding_dim,
        word_dict_size,
        lstm_nr_stacked_layer,
        lstm_h_dim,
        hidden_dim,
        output_dim ;

    dynet::real dropout_rate ; 

protected :
    dynet::Model *m ;
    dynet::Dict input_dict ;
    dynet::Dict output_dict ;
    DictWrapper input_dict_wrapper ;

    Input1 *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;
    CWSTaggingSystem tag_sys ;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(SingleInputModel)

template <typename Archive>
void SingleInputModel::save(Archive &ar, const unsigned version) const
{
    ar & word_dict_size & word_embedding_dim
        & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & input_dict & output_dict ;
    ar & *m ;
}

template <typename Archive>
void SingleInputModel::load(Archive &ar, const unsigned version)
{
    ar & word_dict_size & word_embedding_dim
        & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & input_dict & output_dict ;
    assert(input_dict.size() == word_dict_size && output_dict.size() == output_dim) ;
    build_model_structure() ;
    ar & *m ;
}

} // end of namespcace slnn 


#endif
