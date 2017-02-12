#include "nn_cws_mlp_input1_abstract.h"
namespace slnn{
namespace segmenter{
namespace nn_module{



NnSegmenterInput1MlpAbstract::NnSegmenterInput1MlpAbstract(int argc, char **argv, unsigned seed) : 
    NeuralNetworkCommonInterfaceDynetImpl(argc, argv, seed)
{}

dynet::expr::Expression 
NnSegmenterInput1MlpAbstract::build_training_graph_impl(const std::vector<Index> &charseq, 
    const std::vector<Index> &tagseq)
{
    //clear_cg(); // !! ATTENTION !!
    reset_cg();
    word_expr_layer->new_graph(*get_cg());
    window_expr_generate_layer->new_graph(*get_cg());
    window_expr_processing_layer->new_graph(*get_cg());
    mlp_hidden_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
    
    mlp_hidden_layer->enable_dropout();
    
    unsigned sent_len = charseq.size();

    std::vector<dynet::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // generate window expr
    std::vector<std::vector<dynet::expr::Expression>> input_window_expr_list =
        window_expr_generate_layer->generate_window_expr_list(word_exprs);
    // processing window expr
    std::vector<dynet::expr::Expression> input_exprs = window_expr_processing_layer->process(input_window_expr_list);

    std::vector<dynet::expr::Expression> output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, output_exprs);
    return output_layer->build_output_loss(output_exprs, tagseq);
}

std::vector<Index> 
NnSegmenterInput1MlpAbstract::predict_impl(const std::vector<Index> &charseq)
{

    reset_cg();
    //clear_cg(); // !!!!
    word_expr_layer->new_graph(*get_cg());
    window_expr_generate_layer->new_graph(*get_cg());
    window_expr_processing_layer->new_graph(*get_cg());
    mlp_hidden_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
    mlp_hidden_layer->disable_dropout();

    unsigned sent_len = charseq.size();

    std::vector<dynet::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // generate window expr
    std::vector<std::vector<dynet::expr::Expression>> input_window_expr_list =
        window_expr_generate_layer->generate_window_expr_list(word_exprs);
    // prpcessing window expr
    std::vector<dynet::expr::Expression> input_exprs = 
        window_expr_processing_layer->process(input_window_expr_list);

    std::vector<dynet::expr::Expression> hidden_output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, hidden_output_exprs);
    std::vector<Index> pred_tagseq;
    output_layer->build_output(hidden_output_exprs, pred_tagseq);
    return pred_tagseq;
}

} // end of namespace nn_module
} // end of namespace segmenter
} // end of namespace slnn
