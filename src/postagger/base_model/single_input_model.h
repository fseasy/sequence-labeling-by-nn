#ifndef SLNN_POSTAGGER_BASEMODEL_SINGLE_INPUT_MODEL_H_
#define SLNN_POSTAGGER_BASEMODEL_SINGLE_INPUT_MODEL_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"

namespace slnn{

class SingleInputModel
{
public :
    SingleInputModel() ;
    virtual ~SingleInputModel() ;
    
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0 ;
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


    void set_dynet_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } 

    virtual void save_model(std::ostream &os) = 0 ;
    virtual void load_model(std::istream &is) = 0 ;

public :
    static const std::string UNK_STR;
protected :
    dynet::Model *m ;
    dynet::Dict input_dict ;
    dynet::Dict output_dict ;
    DictWrapper input_dict_wrapper ;

    Input1 *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;

};


} // end of namespcace slnn 


#endif
