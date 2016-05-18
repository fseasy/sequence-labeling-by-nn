#ifndef SLNN_SEGMENTOR_BASEMODEL_SINGLE_INPUT_MODEL_H_
#define SLNN_SEGMENTOR_BASEMODEL_SINGLE_INPUT_MODEL_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
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
    
    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg , 
                                              const IndexSeq &input_seq , const IndexSeq &gold_seq) ;
    virtual void predict(cnn::ComputationGraph &cg ,
                         const IndexSeq &input_seq, IndexSeq &pred_seq) ;

    cnn::Dict& get_input_dict(){ return input_dict ;  } ;
    cnn::Dict& get_output_dict(){ return output_dict ; } ;
    DictWrapper& get_input_dict_wrapper(){ return input_dict_wrapper ; } ;
    cnn::Model *get_cnn_model(){ return m ; } ;
    void set_cnn_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } ;

    virtual void save_model(std::ostream &os) = 0 ;
    virtual void load_model(std::istream &is) = 0 ;

public :
    static const std::string UNK_STR;
protected :
    cnn::Model *m ;
    cnn::Dict input_dict ;
    cnn::Dict output_dict ;
    DictWrapper input_dict_wrapper ;

    Input1 *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;

};

/********* implementation  ********/

const std::string SingleInputModel::UNK_STR = "UNK_STR" ;

SingleInputModel::SingleInputModel()
    :input_layer(nullptr) ,
    bilstm_layer(nullptr) ,
    output_layer(nullptr) ,
    m(nullptr) ,
    input_dict_wrapper(input_dict)
{}

SingleInputModel::~SingleInputModel()
{
    delete input_layer ;
    delete bilstm_layer ;
    delete output_layer ;
    delete m ;
}

cnn::expr::Expression
SingleInputModel::build_loss(cnn::ComputationGraph &cg ,
                             const IndexSeq &input_seq, const IndexSeq &gold_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->set_dropout() ;
    bilstm_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

void 
SingleInputModel::predict(cnn::ComputationGraph &cg,
                          const IndexSeq &input_seq, IndexSeq &pred_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->disable_dropout() ;
    bilstm_layer->start_new_sequence();

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, inputs_exprs) ;
    std::vector<cnn::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_seq) ;
}

} // end of namespcace slnn 


#endif
