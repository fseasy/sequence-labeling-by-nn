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
