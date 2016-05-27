#include "cws_double_classification_model.h"
#include "segmentor/cws_module/cws_output_layer.h"

namespace slnn{

CWSDoubleClassificationModel::CWSDoubleClassificationModel()
    :Input2Model()
{}

CWSDoubleClassificationModel::~CWSDoubleClassificationModel(){}

void CWSDoubleClassificationModel::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(dynamic_dict.is_frozen() && fixed_dict.is_frozen() && tag_dict.is_frozen()) ;

    dynamic_word_dim = var_map["dynamic_word_dim"].as<unsigned>() ;
    fixed_word_dim = var_map["fixed_word_dim"].as<unsigned>() ;
    lstm_nr_stacked_layer = var_map["nr_lstm_stacked_layer"].as<unsigned>() ;
    lstm_x_dim = var_map["lstm_x_dim"].as<unsigned>() ;
    lstm_h_dim = var_map["lstm_h_dim"].as<unsigned>() ;
    hidden_dim = var_map["tag_layer_hidden_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;
    dynamic_dict_size = dynamic_dict.size() ;
    fixed_dict_size = fixed_dict.size() ;
    output_dim = tag_dict.size() ;

}
void CWSDoubleClassificationModel::build_model_structure()
{
    tag_sys.build(tag_dict) ; // init B_ID , M_ID and so on 
    m = new cnn::Model() ;
    input_layer = new Input2(m, dynamic_dict_size, dynamic_word_dim , fixed_dict_size , fixed_word_dim , lstm_x_dim) ;
    bilstm_layer = new BILSTMLayer(m, lstm_nr_stacked_layer, lstm_x_dim, lstm_h_dim, dropout_rate) ;
    output_layer = new CWSSimpleOutput(m, lstm_h_dim, lstm_h_dim, hidden_dim, output_dim , tag_sys) ;
}

void CWSDoubleClassificationModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Single Input Classification Model -----------------\n"
        << "dynamic vocabulary size : " << dynamic_dict_size << " , dimension : " << dynamic_word_dim << "\n"
        << "fixed vocabulary size : " << fixed_dict_size << " , dimension : " << fixed_word_dim << "\n"
        << "bi-lstm x dim : " << lstm_x_dim << " , h dim : " << lstm_h_dim
        << " , stacked layer num : " << lstm_nr_stacked_layer << "\n"
        << "tag hidden layer dim : " << hidden_dim << "\n"
        << "output dim : " << output_dim ;
}

template <typename Archive>
void CWSDoubleClassificationModel::save(Archive &ar, const unsigned version) const
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    ar & dynamic_dict_size & dynamic_word_dim
        & fixed_dict_size & fixed_word_dim
        & lstm_x_dim & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & dynamic_dict & fixed_dict & tag_dict ;
    boost::serialization::base_object<Input2Model>(*this) ;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

template <typename Archive>
void CWSDoubleClassificationModel::load(Archive &ar, const unsigned version)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    ar & dynamic_dict_size & dynamic_word_dim
        & fixed_dict_size & fixed_word_dim
        & lstm_x_dim & lstm_h_dim & lstm_nr_stacked_layer
        & hidden_dim & output_dim
        & dropout_rate ;
    ar & dynamic_dict & fixed_dict & tag_dict ;
    assert(dynamic_dict.size() == dynamic_dict_dize && fixed_dict.size() == fixed_dict_size &&
           tag_dict.size() == output_dim) ;
    build_model_structure() ;
    boost::serialization::base_object<Input2Model>(*this) ;
}

}
