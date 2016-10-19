#ifndef SLNN_SEGMENTOR_BASEMODEL_INPUT2_MODEL_H_
#define SLNN_SEGMENTOR_BASEMODEL_INPUT2_MODEL_H_

#include <iostream>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"
#include "segmenter/cws_module/cws_tagging_system.h"
namespace slnn{

class Input2Model
{
    friend class boost::serialization::access ;

public :
    Input2Model() ;
    virtual ~Input2Model() ;
    Input2Model(const Input2Model &) = delete ;
    Input2Model& operator=(const Input2Model &) = delete ;
    
    void set_fixed_word_dict_size_and_embedding(unsigned fixed_word_dict_size, unsigned fixed_word_dim);
    virtual void set_model_param(const boost::program_options::variables_map &var_map);

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;
    
    virtual dynet::expr::Expression  build_loss(dynet::ComputationGraph &cg, 
                                              const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent, 
                                              const IndexSeq &gold_tag_seq) ;
    virtual void predict(dynet::ComputationGraph &cg ,
                         const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent,
                         IndexSeq &pred_tag_seq) ;

    dynet::LookupParameter  get_fixed_lookup_param(){ return input_layer->fixed_lookup_param ; }
    dynet::Dict& get_dynamic_dict(){ return dynamic_dict ;  } 
    dynet::Dict& get_fixed_dict(){ return fixed_dict ; }
    dynet::Dict& get_tag_dict(){ return tag_dict ; } 
    DictWrapper& get_dynamic_dict_wrapper(){ return dynamic_dict_wrapper ; } 
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

    unsigned dynamic_word_dim,
        fixed_word_dim,
        dynamic_dict_size,
        fixed_dict_size,
        lstm_nr_stacked_layer,
        lstm_x_dim,
        lstm_h_dim,
        hidden_dim,
        output_dim ;

    dynet::real dropout_rate ; 
protected :
    dynet::Model *m ;
    dynet::Dict dynamic_dict;
    dynet::Dict fixed_dict;
    dynet::Dict tag_dict;
    DictWrapper dynamic_dict_wrapper;

    Input2 *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;
    CWSTaggingSystem tag_sys ;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Input2Model)

template <typename Archive>
void Input2Model::save(Archive &ar, const unsigned version) const
{
    ar & dynamic_dict_size & dynamic_word_dim
        & fixed_dict_size & fixed_word_dim
        & lstm_x_dim & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & dynamic_dict & fixed_dict & tag_dict ;
    ar & *m ;
}

template <typename Archive>
void Input2Model::load(Archive &ar, const unsigned version)
{
    ar & dynamic_dict_size & dynamic_word_dim
        & fixed_dict_size & fixed_word_dim
        & lstm_x_dim & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & dynamic_dict & fixed_dict & tag_dict ;
    assert(dynamic_dict.size() == dynamic_dict_size && fixed_dict.size() == fixed_dict_size &&
           tag_dict.size() == output_dim) ;
    build_model_structure() ;
    ar & *m ;
}

} // end of namespcace slnn 

#endif
