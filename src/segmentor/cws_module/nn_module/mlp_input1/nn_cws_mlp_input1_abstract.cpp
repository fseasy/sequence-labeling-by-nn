#include "nn_cws_mlp_input1_abstract.h"
namespace slnn{
namespace segmentor{
namespace nn_module{



NnSegmentorInput1Abstract::NnSegmentorInput1Abstract(int argc, char **argv, unsigned seed) : 
    NeuralNetworkCommonInterfaceCnnImpl(argc, argv, seed)
{}

cnn::expr::Expression 
NnSegmentorInput1Abstract::build_training_graph_impl(const std::vector<Index> &charseq, 
    const std::vector<Index> &tagseq)
{
    clear_cg(); // !! ATTENTION !!
    word_expr_layer->new_graph(*get_cg());
    window_expr_generate_layer->new_graph(*get_cg());
    mlp_hidden_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;
    
    mlp_hidden_layer->enable_dropout();
    
    unsigned sent_len = charseq.size();

    std::vector<cnn::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // generate window expr(context) using concatenate
    std::vector<cnn::expr::Expression> input_exprs = window_expr_generate_layer->generate_window_expr_by_concatenating(word_exprs);

    std::vector<cnn::expr::Expression> output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, output_exprs);
    return output_layer->build_output_loss(output_exprs, tagseq);
}

std::vector<Index> 
NnSegmentorInput1Abstract::predict_impl(const std::vector<Index> &charseq)
{
    clear_cg(); // !!!!
    word_expr_layer->new_graph(*get_cg());
    window_expr_generate_layer->new_graph(*get_cg());
    mlp_hidden_layer->new_graph(*get_cg());
    output_layer->new_graph(*get_cg()) ;

    mlp_hidden_layer->disable_dropout();

    unsigned sent_len = charseq.size();

    std::vector<cnn::expr::Expression> word_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(charseq, word_exprs);
    // generate window expr(context) using concatenate
    std::vector<cnn::expr::Expression> input_exprs = window_expr_generate_layer->generate_window_expr_by_concatenating(word_exprs);

    std::vector<cnn::expr::Expression> hidden_output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, hidden_output_exprs);
    std::vector<Index> pred_tagseq;
    output_layer->build_output(hidden_output_exprs, pred_tagseq);
    return pred_tagseq;
}

} // end of namespace nn_module
} // end of namespace segmentor
} // end of namespace slnn
