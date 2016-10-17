#include "cws_single_classification_model.h"
#include "segmenter/cws_module/cws_output_layer.h"

namespace slnn{

CWSSingleClassificationModel::CWSSingleClassificationModel()
    :SingleInputModel() ,
    word_dict(input_dict) ,
    tag_dict(output_dict)
{}

CWSSingleClassificationModel::~CWSSingleClassificationModel(){}

void CWSSingleClassificationModel::set_model_param(const boost::program_options::variables_map &var_map)
{
    CWSSingleClassificationModel::SingleInputModel::set_model_param(var_map);
}
void CWSSingleClassificationModel::build_model_structure()
{
    tag_sys.build(tag_dict) ; // init B_ID , M_ID and so on 
    m = new cnn::Model() ;
    input_layer = new Input1(m, word_dict_size, word_embedding_dim) ;
    bilstm_layer = new BILSTMLayer(m, lstm_nr_stacked_layer, word_embedding_dim, lstm_h_dim, dropout_rate) ;
    output_layer = new CWSSimpleOutput(m, lstm_h_dim, lstm_h_dim, hidden_dim, output_dim , tag_sys) ;
}

void CWSSingleClassificationModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Single Input Classification Model -----------------\n"
        << "vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "bi-lstm x dim : " << word_embedding_dim << " , h dim : " << lstm_h_dim
        << " , stacked layer num : " << lstm_nr_stacked_layer << "\n"
        << "tag hidden layer dim : " << hidden_dim << "\n"
        << "dropout rate : " << dropout_rate << "\n"
        << "output dim : " << output_dim ;
}

}
