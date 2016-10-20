#include "single_input_model.h"

namespace slnn{

const std::string SingleInputModel::UNK_STR = "UNK_STR" ;

SingleInputModel::SingleInputModel()
    :m(nullptr) ,
    input_dict_wrapper(input_dict) ,
    input_layer(nullptr) ,
    bilstm_layer(nullptr) ,
    output_layer(nullptr) 
{}

SingleInputModel::~SingleInputModel()
{
    delete input_layer ;
    delete bilstm_layer ;
    delete output_layer ;
    delete m ;
}

dynet::expr::Expression
SingleInputModel::build_loss(dynet::ComputationGraph &cg ,
                             const IndexSeq &input_seq, const IndexSeq &gold_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->set_dropout() ;
    bilstm_layer->start_new_sequence() ;

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, inputs_exprs) ;

    std::vector<dynet::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_seq) ;
}

void 
SingleInputModel::predict(dynet::ComputationGraph &cg,
                          const IndexSeq &input_seq, IndexSeq &pred_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->disable_dropout() ;
    bilstm_layer->start_new_sequence();

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(input_seq, inputs_exprs) ;
    std::vector<dynet::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_seq) ;
}

} // end of namespace slnn
