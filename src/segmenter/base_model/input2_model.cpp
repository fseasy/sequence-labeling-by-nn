#include "input2_model.h"

namespace slnn{

const std::string Input2Model::UNK_STR = "UNK_STR" ;

Input2Model::Input2Model()
    :m(nullptr) ,
    dynamic_dict_wrapper(dynamic_dict) ,
    input_layer(nullptr) ,
    bilstm_layer(nullptr) ,
    output_layer(nullptr) 
{}

Input2Model::~Input2Model()
{
    delete input_layer ;
    delete bilstm_layer ;
    delete output_layer ;
    delete m ;
}

void Input2Model::set_fixed_word_dict_size_and_embedding(unsigned fdict_sz, unsigned fword_dim)
{
    fixed_dict_size = fdict_sz;
    fixed_word_dim = fword_dim;
}

void Input2Model::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(dynamic_dict.is_frozen() && fixed_dict.is_frozen() && tag_dict.is_frozen()) ;

    dynamic_word_dim = var_map["dynamic_word_dim"].as<unsigned>() ;
    lstm_nr_stacked_layer = var_map["nr_lstm_stacked_layer"].as<unsigned>() ;
    lstm_x_dim = var_map["lstm_x_dim"].as<unsigned>() ;
    lstm_h_dim = var_map["lstm_h_dim"].as<unsigned>() ;
    hidden_dim = var_map["tag_layer_hidden_dim"].as<unsigned>() ;

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;

    dynamic_dict_size = dynamic_dict.size() ;
    assert(fixed_dict_size == fixed_dict.size()) ;
    output_dim = tag_dict.size() ;
}

cnn::expr::Expression
Input2Model::build_loss(cnn::ComputationGraph &cg ,
                        const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent, 
                        const IndexSeq &gold_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->set_dropout() ;
    bilstm_layer->start_new_sequence() ;

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(dynamic_sent, fixed_sent , inputs_exprs) ;

    std::vector<cnn::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

void 
Input2Model::predict(cnn::ComputationGraph &cg,
                     const IndexSeq &dynamic_sent, const IndexSeq &fixed_sent,
                     IndexSeq &pred_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->disable_dropout() ;
    bilstm_layer->start_new_sequence();

    std::vector<cnn::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(dynamic_sent, fixed_sent, inputs_exprs) ;
    std::vector<cnn::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_seq) ;
}



} // end of namespace slnn
