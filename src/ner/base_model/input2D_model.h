#ifndef SLNN_NER_BASEMODEL_INPUT2D_MODEL_H_
#define SLNN_NER_BASEMODEL_INPUT2D_MODEL_H_

#include <iostream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"

namespace slnn{

class Input2DModel
{
public :
    Input2DModel() ;
    virtual ~Input2DModel() ;
    
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0 ;
    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;
    
    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
                                              const IndexSeq &words_seq, const IndexSeq &postag_seq,
                                              const IndexSeq &gold_ner_seq) ;
    virtual void predict(cnn::ComputationGraph &cg ,
                         const IndexSeq &words_seq, const IndexSeq &postag_seq, 
                         IndexSeq &pred_ner_seq) ;

    cnn::Dict& get_word_dict(){ return word_dict ;  } 
    cnn::Dict& get_postag_dict() { return postag_dict ; }
    cnn::Dict& get_ner_dict(){ return ner_dict ; } 
    DictWrapper& get_word_dict_wrapper(){ return word_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } ;


    void set_cnn_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } 

    virtual void save_model(std::ostream &os) = 0 ;
    virtual void load_model(std::istream &is) = 0 ;

public :
    static const std::string UNK_STR;
protected :
    cnn::Model *m ;
    cnn::Dict word_dict ;
    cnn::Dict postag_dict ;
    cnn::Dict ner_dict ;
    DictWrapper word_dict_wrapper ;

    Input2D *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;

};


} // end of namespcace slnn 


#endif
