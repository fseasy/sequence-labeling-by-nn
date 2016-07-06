#ifndef POS_BASE_MODEL_BAREINPUT1_FEATURE2OUTPUT_LAYER_NONLINEAR_MODEL_HPP_
#define POS_BASE_MODEL_BAREINPUT1_FEATURE2OUTPUT_LAYER_NONLINEAR_MODEL_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/split_member.hpp>

#include "cnn/cnn.h"
#include "cnn/dict.h"

#include "utils/typedeclaration.h"
#include "bareinput1_f2o_no_merge_model.hpp"
namespace slnn{

template<typename RNNDerived>
class BareInput1F2ONonlinearModel : public BareInput1F2OModel<RNNDerived>
{
    friend class boost::serialization::access;

public:
    BareInput1F2ONonlinearModel();
    virtual ~BareInput1F2ONonlinearModel();

    virtual void set_model_param(const boost::program_options::variables_map &var_map);

    virtual void build_model_structure() = 0 ;
    virtual void print_model_info() = 0 ;

    virtual cnn::expr::Expression  build_loss(cnn::ComputationGraph &cg,
        const IndexSeq &input_seq, 
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        const IndexSeq &gold_seq) override ;
    virtual void predict(cnn::ComputationGraph &cg ,
        const IndexSeq &input_seq, 
        const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
        IndexSeq &pred_seq) override;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned version);

protected:

    DenseLayer *pos_feature_hidden_layer;

public:
    unsigned pos_feature_hidden_layer_dim ; 
private :
    NonLinearFunc *pos_feature_hidden_layer_nonlinear_func;
    std::string nonlinear_func_name;
};


template<typename RNNDerived>
BareInput1F2ONonlinearModel<RNNDerived>::BareInput1F2ONonlinearModel() 
    :BareInput1F2OModel<RNNDerived>(),
    pos_feature_hidden_layer(nullptr)
{}

template <typename RNNDerived>
BareInput1F2ONonlinearModel<RNNDerived>::~BareInput1F2ONonlinearModel()
{
    delete pos_feature_hidden_layer;
}

template <typename RNNDerived>
void BareInput1F2ONonlinearModel<RNNDerived>::set_model_param(const boost::program_options::variables_map &var_map)
{
    pos_feature_hidden_layer_dim = var_map["pos_feature_hidden_layer_dim"].as<unsigned>();
    nonlinear_func_name = var_map["pos_feature_hidden_layer_nonlinear_func"].as<std::string>();
    if( nonlinear_func_name == "rectify" ){ pos_feature_hidden_layer_nonlinear_func = &cnn::expr::rectify; }
    else if( nonlinear_func_name == "tanh" ){ pos_feature_hidden_layer_nonlinear_func = &cnn::expr::tanh; }
    else{ throw std::runtime_error(std::string("unsupported nonlinear func for ") + nonlinear_func_name); }

    BareInput1F2ONonlinearModel<RNNDerived>::BareInput1F2OModel::set_model_param(var_map);
    this->softmax_input_dim = this->rnn_h_dim * 2 + pos_feature_hidden_layer_dim; // base class's value is not suitable !
}


template<typename RNNDerived>
cnn::expr::Expression BareInput1F2ONonlinearModel<RNNDerived>::build_loss(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq, 
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    const IndexSeq &gold_seq)
{
    this->pos_feature_layer->new_graph(cg);
    pos_feature_hidden_layer->new_graph(cg);
    this->input_layer->new_graph(cg) ;
    this->birnn_layer->new_graph(cg) ;
    this->output_layer->new_graph(cg) ;

    this->birnn_layer->set_dropout() ;
    this->birnn_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> inputs_exprs ;
    this->input_layer->build_inputs(input_seq, inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    this->birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;

    std::vector<cnn::expr::Expression> feature_exprs;
    this->pos_feature_layer->build_feature_exprs(features_gp_seq, feature_exprs);
    for( size_t i = 0; i < input_seq.size(); ++i )
    {
        feature_exprs.at(i) = (*pos_feature_hidden_layer_nonlinear_func)(
            pos_feature_hidden_layer->build_graph(feature_exprs.at(i))
        );
    }
    return this->output_layer->build_output_loss(std::vector<std::vector<cnn::expr::Expression> *>({ &l2r_exprs, &r2l_exprs, &feature_exprs }),
        gold_seq) ;
}

template<typename RNNDerived>
void BareInput1F2ONonlinearModel<RNNDerived>::predict(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    IndexSeq &pred_seq)
{
    this->pos_feature_layer->new_graph(cg);
    pos_feature_hidden_layer->new_graph(cg);
    this->input_layer->new_graph(cg) ;
    this->birnn_layer->new_graph(cg) ;
    this->output_layer->new_graph(cg) ;

    this->birnn_layer->disable_dropout() ;
    this->birnn_layer->start_new_sequence();

    std::vector<cnn::expr::Expression> inputs_exprs ;
    this->input_layer->build_inputs(input_seq, inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
        r2l_exprs ;
    this->birnn_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;

    std::vector<cnn::expr::Expression> feature_exprs;
    this->pos_feature_layer->build_feature_exprs(features_gp_seq, feature_exprs);
    
    for( size_t i = 0; i < input_seq.size(); ++i )
    {
        feature_exprs.at(i) = (*pos_feature_hidden_layer_nonlinear_func)(
            pos_feature_hidden_layer->build_graph(feature_exprs.at(i))
        );
    }
    
    this->output_layer->build_output(std::vector<std::vector<cnn::expr::Expression> *>({ &l2r_exprs, &r2l_exprs, &feature_exprs }),
        pred_seq) ;
}

template <typename RNNDerived>template< typename Archive>
void BareInput1F2ONonlinearModel<RNNDerived>::serialize(Archive & ar, const unsigned version)
{
    ar & pos_feature_hidden_layer_dim 
       & nonlinear_func_name ;
    if( nonlinear_func_name == "rectify" ){ pos_feature_hidden_layer_nonlinear_func = &cnn::expr::rectify; }
    else if( nonlinear_func_name == "tanh" ){ pos_feature_hidden_layer_nonlinear_func = &cnn::expr::tanh; }
    else{ throw std::runtime_error(std::string("unsupported nonlinear func for ") + nonlinear_func_name); }
    ar & boost::serialization::base_object<BareInput1F2OModel<RNNDerived>>(*this);
}

} // end of namespace slnn
#endif
