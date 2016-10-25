#include "nn_window_expr_processing_layer.h"
namespace slnn{
namespace segmenter{
namespace nn_module{
namespace experiment{

/***********
* Window Expression Concatenate layer
***********/

WindowExprConcatLayer::WindowExprConcatLayer(unsigned unit_embedding_dim, unsigned window_sz)
    :output_dim(unit_embedding_dim * window_sz)
{}

/*************
* Window Expression Sum(average) Layer
*************/

WindowExprSumLayer::WindowExprSumLayer(unsigned unit_embedding_dim)
    :output_dim(unit_embedding_dim)
{}

/*************
* Window Expression Bigram Layer
*************/

WindowExprBigramLayer::WindowExprBigramLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz)
    : Tl_param(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    Tr_param(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    b_param(dynet_model->add_parameters({ unit_embedding_dim })),
    cached_c_expr_list(window_sz - 1),
    output_dim(unit_embedding_dim)
{
    assert(window_sz > 0);
}


std::vector<dynet::expr::Expression>
WindowExprBigramLayer::process(const std::vector<std::vector<dynet::expr::Expression>> &window_expr_list)
{
    unsigned len = window_expr_list.size();
    std::vector<dynet::expr::Expression> bigram_expr_list(len);
    // the first cached c expression
    for( unsigned i = 0; i < cached_c_expr_list.size(); ++i )
    {
        cached_c_expr_list[i] = dynet::expr::tanh(
            dynet::expr::affine_transform({
            b_expr,
            Tl_expr, window_expr_list[0][i],
            Tr_expr, window_expr_list[0][i + 1]
        })
        );
    }
    bigram_expr_list[0] = dynet::expr::sum(cached_c_expr_list);
    // the continues expression
    // -> rolling
    // bigram: current: [AB, BC, CD, DE,] EF, FG, ...
    //         next   : AB, [BC, CD, DE, EF,] EG, ...
    // input window: current: [A,B,C,D,E],[B,C,D,E,F]
    for( unsigned j = 1; j < len; ++j )
    {
        cached_c_expr_list.pop_front();
        dynet::expr::Expression next_expr = dynet::expr::tanh(
            dynet::expr::affine_transform({
            b_expr,
            Tl_expr, *(window_expr_list[j].end() - 2),
            Tr_expr, window_expr_list[j].back()
        })
        );
        cached_c_expr_list.push_back(next_expr);
        bigram_expr_list[j] = dynet::expr::sum(cached_c_expr_list);
    }
    return bigram_expr_list;
}

/***********
 * create window expression processing layer
 *********/
std::shared_ptr<WindowExprProcessingLayerAbstract> 
create_window_expr_processing_layer(const std::string& processing_method, 
    dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz)
{
    std::string name(processing_method);
    for( auto &c : name ){ c = ::tolower(c); }
    if( name == "concat" || name == "concatenate" )
    {
        return std::shared_ptr<WindowExprProcessingLayerAbstract>(
            new WindowExprConcatLayer(unit_embedding_dim, window_sz)
            );
    }
    else if( name == "sum" || name == "avg" || name == "average" )
    {
        return std::shared_ptr<WindowExprProcessingLayerAbstract>(
            new WindowExprSumLayer(unit_embedding_dim)
            );
    }
    else if( name == "bigram" || name == "bigram-cnn" || name == "bigram_cnn" )
    {
        return std::shared_ptr<WindowExprProcessingLayerAbstract>(
            new WindowExprBigramLayer(dynet_model, unit_embedding_dim, window_sz)
            );
    }
    else
    { 
        throw std::invalid_argument("unsupported window expression processing method: '" + processing_method + "'\n"
            "supporting list: concat, avg, bigram\n");
    }
}

} // end of namespace experiment
} // end of namespace nn_module
} // end of namespace segmenter
} // end of namespace slnn
