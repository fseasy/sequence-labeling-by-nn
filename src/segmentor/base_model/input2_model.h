#ifndef SLNN_SEGMENTOR_BASEMODEL_INPUT2_MODEL_H_
#define SLNN_SEGMENTOR_BASEMODEL_INPUT2_MODEL_H_

#include <iostream>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"
#include "utils/typedeclaration.h"
#include "utils/dict_wrapper.hpp"
#include "modelmodule/hyper_layers.h"
#include "segmentor/cws_module/cws_tagging_system.h"
namespace slnn{

class Input2Model
{
    friend class boost::serialization::access ;

public :
    Input2Model() ;
    virtual ~Input2Model() ;
    Input2Model(const Input2Model &) = delete ;
    Input2Model& operator=(const Input2Model &) = delete ;
    
    virtual void set_model_param(const boost::program_options::variables_map &var_map) = 0 ;
    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;
    
    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg, 
                                              const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent, 
                                              const IndexSeq &gold_tag_seq) ;
    virtual void predict(cnn::ComputationGraph &cg ,
                         const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent,
                         IndexSeq &pred_tag_seq) ;

    cnn::Dict& get_dynamic_dict(){ return dynamic_dict ;  } 
    cnn::Dict& get_fixed_dict(){ return fixed_dict ; }
    cnn::Dict& get_tag_dict(){ return tag_dict ; } 
    DictWrapper& get_dynamic_dict_wrapper(){ return dynamic_dict_wrapper ; } 
    cnn::Model *get_cnn_model(){ return m ; } ;
    CWSTaggingSystem& get_tag_sys(){ return tag_sys ; }

    void set_cnn_model(std::istream &mis){ boost::archive::text_iarchive ti(mis) ; ti >> *m ; } 

    virtual void save_model(std::ostream &os) = 0 ;
    virtual void load_model(std::istream &is) = 0 ;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned versoin){ ar & *m ; }

public :
    static const std::string UNK_STR;
protected :
    cnn::Model *m ;
    cnn::Dict dynamic_dict;
    cnn::Dict fixed_dict;
    cnn::Dict tag_dict;
    DictWrapper dynamic_dict_wrapper;

    Input2 *input_layer ;
    BILSTMLayer *bilstm_layer ;
    OutputBase *output_layer ;
    CWSTaggingSystem tag_sys ;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Input2Model)

} // end of namespcace slnn 


#endif
