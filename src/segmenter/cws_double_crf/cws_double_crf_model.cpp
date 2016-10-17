#include "cws_double_crf_model.h"
#include "segmenter/cws_module/cws_output_layer.h"

namespace slnn{

CWSDoubleCRFModel::CWSDoubleCRFModel()
    :Input2Model()
{}

CWSDoubleCRFModel::~CWSDoubleCRFModel(){}

void CWSDoubleCRFModel::set_model_param(const boost::program_options::variables_map &var_map)
{
    tag_dim = var_map["tag_dim"].as<unsigned>();
    CWSDoubleCRFModel::Input2Model::set_model_param(var_map);
}
void CWSDoubleCRFModel::build_model_structure()
{
    tag_sys.build(tag_dict) ; // init B_ID , M_ID and so on 
    m = new cnn::Model() ;
    input_layer = new Input2(m, dynamic_dict_size, dynamic_word_dim , fixed_dict_size , fixed_word_dim , lstm_x_dim) ;
    bilstm_layer = new BILSTMLayer(m, lstm_nr_stacked_layer, lstm_x_dim, lstm_h_dim, dropout_rate) ;
    output_layer = new CWSCRFOutput(m, tag_dim, lstm_h_dim, lstm_h_dim, hidden_dim, output_dim , dropout_rate, tag_sys) ;
}

void CWSDoubleCRFModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Input2 CRF Model -----------------\n"
        << "dynamic vocabulary size : " << dynamic_dict_size << " , dimension : " << dynamic_word_dim << "\n"
        << "fixed vocabulary size : " << fixed_dict_size << " , dimension : " << fixed_word_dim << "\n"
        << "tag dict size : " << output_dim << " , dimension : " << tag_dim << "\n"
        << "dropout rate : " << dropout_rate << "\n"
        << "bi-lstm x dim : " << lstm_x_dim << " , h dim : " << lstm_h_dim
        << " , stacked layer num : " << lstm_nr_stacked_layer << "\n"
        << "tag hidden layer dim : " << hidden_dim << "\n"
        << "dropout rate : " << dropout_rate << "\n"
        << "output dim : " << output_dim ;
}

}
