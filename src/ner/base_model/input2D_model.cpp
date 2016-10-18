#include "input2D_model.h"

namespace slnn{

const std::string Input2DModel::UNK_STR = "UNK_STR" ;

Input2DModel::Input2DModel()
    :m(nullptr) ,
    word_dict_wrapper(word_dict) ,
    input_layer(nullptr) ,
    bilstm_layer(nullptr) ,
    output_layer(nullptr) 
{}

Input2DModel::~Input2DModel()
{
    delete input_layer ;
    delete bilstm_layer ;
    delete output_layer ;
    delete m ;
}

dynet::expr::Expression
Input2DModel::build_loss(dynet::ComputationGraph &cg,
                         const IndexSeq &words_seq, const IndexSeq &postag_seq,
                         const IndexSeq &gold_ner_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->set_dropout() ;
    bilstm_layer->start_new_sequence() ;

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(words_seq, postag_seq , inputs_exprs) ;

    std::vector<dynet::expr::Expression> l2r_exprs,
        r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    return output_layer->build_output_loss(l2r_exprs, r2l_exprs, gold_ner_seq) ;
}

void 
Input2DModel::predict(dynet::ComputationGraph &cg,
                      const IndexSeq &words_seq, const IndexSeq &postag_seq , 
                      IndexSeq &pred_ner_seq)
{
    input_layer->new_graph(cg) ;
    bilstm_layer->new_graph(cg) ;
    output_layer->new_graph(cg) ;

    bilstm_layer->disable_dropout() ;
    bilstm_layer->start_new_sequence();

    std::vector<dynet::expr::Expression> inputs_exprs ;
    input_layer->build_inputs(words_seq , postag_seq, inputs_exprs) ;
    std::vector<dynet::expr::Expression> l2r_exprs,
                                       r2l_exprs ;
    bilstm_layer->build_graph(inputs_exprs, l2r_exprs, r2l_exprs) ;
    output_layer->build_output(l2r_exprs, r2l_exprs , pred_ner_seq) ;
}

} // end of namespace slnn
