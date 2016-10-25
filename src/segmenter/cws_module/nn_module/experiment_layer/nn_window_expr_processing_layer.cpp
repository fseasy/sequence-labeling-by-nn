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
    assert(window_sz > 1);
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
 * Window Expression Bigram by Concat Layer
 **********/

WindowExprBigramConcatLayer::WindowExprBigramConcatLayer(dynet::Model *dynet_model, unsigned unit_embedding_dim, unsigned window_sz)
    :Tl_param(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    Tr_param(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    b_param(dynet_model->add_parameters({ unit_embedding_dim })),
    cached_c_expr_list(window_sz - 1),
    output_dim(unit_embedding_dim * (window_sz - 1))
{
    assert(window_sz > 1);
}

std::vector<dynet::expr::Expression>
WindowExprBigramConcatLayer::process(const std::vector<std::vector<dynet::expr::Expression>> &window_expr_list)
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
    bigram_expr_list[0] = dynet::expr::concatenate(cached_c_expr_list);
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
        bigram_expr_list[j] = dynet::expr::concatenate(cached_c_expr_list);
    }
    return bigram_expr_list;
}

/**********
 * Window Expr Attention 1 Layer
 **********/
WindowExprAttention1Layer::WindowExprAttention1Layer(dynet::Model *dynet_model,
    unsigned unit_embedding_dim, unsigned window_sz)
    : W(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    U(dynet_model->add_parameters({ unit_embedding_dim, unit_embedding_dim })),
    v(dynet_model->add_parameters({ 1, unit_embedding_dim })),
    output_dim(unit_embedding_dim * window_sz),
    window_sz(window_sz)
{}


std::vector<dynet::expr::Expression>
WindowExprAttention1Layer::process(const std::vector<std::vector<dynet::expr::Expression>>& window_expr_list)
{
    // init first 
    unsigned half_sz = window_sz / 2;
    std::deque<dynet::expr::Expression> left_context_mul(half_sz);
    std::deque<dynet::expr::Expression> right_context_mul(half_sz);
    for( unsigned i = 0; i < half_sz; ++i )
    {
        left_context_mul[i] = W_expr * window_expr_list[0][i];
        right_context_mul[i] = W_expr * window_expr_list[0][half_sz + i + 1];
    }
    dynet::expr::Expression center_mul = U_expr * window_expr_list[0][half_sz];
    std::vector<dynet::expr::Expression> score_list(window_sz - 1);
    for( unsigned i = 0; i < half_sz; ++i )
    {
        score_list[i] = v_expr * dynet::expr::tanh(left_context_mul[i] + center_mul);
        score_list[i + half_sz] = v_expr * dynet::expr::tanh(right_context_mul[i] + center_mul);
    }
    dynet::expr::Expression weight_list = dynet::expr::softmax(dynet::expr::concatenate(score_list));
    std::vector<dynet::expr::Expression> result_expr_window(window_sz);

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
