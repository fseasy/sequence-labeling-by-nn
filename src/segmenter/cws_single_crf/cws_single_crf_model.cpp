#include "cws_single_crf_model.h"
#include "segmenter/cws_module/cws_output_layer.h"

namespace slnn{

CWSSingleCRFModel::CWSSingleCRFModel()
    :SingleInputModel() ,
    word_dict(input_dict) ,
    tag_dict(output_dict)
{}

CWSSingleCRFModel::~CWSSingleCRFModel(){}

void CWSSingleCRFModel::set_model_param(const boost::program_options::variables_map &var_map)
{
    tag_embedding_dim = var_map["tag_embedding_dim"].as<unsigned>() ;
    CWSSingleCRFModel::SingleInputModel::set_model_param(var_map);
}
void CWSSingleCRFModel::build_model_structure()
{
    tag_sys.build(tag_dict) ; // init B_ID , M_ID and so on 
    m = new dynet::Model() ;
    input_layer = new Input1(m, word_dict_size, word_embedding_dim) ;
    bilstm_layer = new BILSTMLayer(m, lstm_nr_stacked_layer, word_embedding_dim, lstm_h_dim, dropout_rate) ;
    output_layer = new CWSCRFOutput(m, tag_embedding_dim, lstm_h_dim, lstm_h_dim, hidden_dim, output_dim, dropout_rate, tag_sys) ; 
}

void CWSSingleCRFModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Single Input CRF Model -----------------\n"
        << "vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "tag dict size : " << output_dim << " with dimension : " << tag_embedding_dim << "\n"
        << "bi-lstm x dim : " << word_embedding_dim << " , h dim : " << lstm_h_dim
        << " , stacked layer num : " << lstm_nr_stacked_layer << "\n"
        << "dropout rate : " << dropout_rate << "\n"
        << "tag hidden layer dim : " << hidden_dim << "\n"
        << "output dim : " << output_dim ;
}

}
