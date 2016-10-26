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
    unsigned len = window_expr_list.size();
    std::vector<dynet::expr::Expression> result_expr_list(len);
    // init first 
    unsigned half_sz = window_sz / 2;
    // - [A, B, C, D, E], left => [A, B], right => [D, E], center => C
    // - use center word to attend the context. => C to attend the [A, B, D, E]
    // - do W*A, W*B, W*D, W*E
    std::deque<dynet::expr::Expression> left_context_mul(half_sz);
    std::deque<dynet::expr::Expression> right_context_mul(half_sz);
    for( unsigned i = 0; i < half_sz; ++i )
    {
        left_context_mul[i] = W_expr * window_expr_list[0][i];
        right_context_mul[i] = W_expr * window_expr_list[0][half_sz + i + 1];
    }
    // - do U*C
    dynet::expr::Expression center_mul = U_expr * window_expr_list[0][half_sz];
    // - do score<context, center> = v * tanh(W*context + U*center)
    // - context => [A, B, D, E], center => [C,] 
    std::vector<dynet::expr::Expression> score_list(window_sz - 1);
    for( unsigned i = 0; i < half_sz; ++i )
    {
        score_list[i] = v_expr * dynet::expr::tanh(left_context_mul[i] + center_mul);
        score_list[i + half_sz] = v_expr * dynet::expr::tanh(right_context_mul[i] + center_mul);
    }
    // - do softmax on score.
    // - score => weight
    dynet::expr::Expression weight_list = dynet::expr::softmax(dynet::expr::concatenate(score_list));
    // - weight the context.
    // - weight_A * A, wegith_B * B, weight_D * D, weight_E * E
    std::vector<dynet::expr::Expression> result_expr_window(window_sz);
    for( unsigned i = 0; i < half_sz; ++i )
    {
        result_expr_window[i] = window_expr_list[0][i] * dynet::expr::pick(weight_list, i);
        // result_expr_window, window_expr_list has `window_sz` expressions, including the center word expression
        // weight_list only have `left_sz + right_sz` expressions, excluding the center word expression
        result_expr_window[i + half_sz + 1] = 
            window_expr_list[0][i + half_sz + 1] * dynet::expr::pick(weight_list, i + half_sz);
    }
    // - center = center
    result_expr_window[half_sz] = window_expr_list[0][half_sz];
    // - concatenate
    result_expr_list[0] = dynet::expr::concatenate(result_expr_window);

    // do continues(using the previous multiply result.)
    for( unsigned i = 1; i < len; ++i )
    {
        // - do context multipilication.
        // - A, B, C, D, E, F => previous window [(A, B), C, (D, E)], next window [(B, C), D, (E, F)]
        // - left context slide one and `half_sz - 1` in; right context slide one and `window_sz - 1` in
        left_context_mul.pop_front();
        left_context_mul.push_back(W_expr * window_expr_list[i][half_sz - 1]); // the first out, the (half_sz-1) in
        right_context_mul.pop_front();
        right_context_mul.push_back(W_expr * window_expr_list[i].back());
        // - center mul
        center_mul = U_expr * window_expr_list[i][half_sz];
        // - score
        for( unsigned j = 0; j < half_sz - 1; ++j )
        {
            score_list[j] = v_expr * dynet::expr::tanh(left_context_mul[j] + center_mul);
            score_list[j + half_sz] = v_expr * dynet::expr::tanh(right_context_mul[j] + center_mul);
        }
        // - softmax to calc weight
        weight_list = dynet::expr::softmax(dynet::expr::concatenate(score_list));
        // - weight the context
        for( unsigned window_idx = 0; window_idx < half_sz; ++window_idx )
        {
            result_expr_window[window_idx] = window_expr_list[i][window_idx] * dynet::expr::pick(weight_list, window_idx);
            result_expr_window[window_idx + half_sz + 1] =
                window_expr_list[i][window_idx + half_sz + 1] * dynet::expr::pick(weight_list, window_idx + half_sz);
        }
        result_expr_window[half_sz] = window_expr_list[i][half_sz];
        // - concatenate
        result_expr_list[i] = dynet::expr::concatenate(result_expr_window);
    }
    return result_expr_list;
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
    else if( name == "bigram_concat" || name == "bigram-concat" || name == "bigramconcat" )
    {
        return std::shared_ptr<WindowExprProcessingLayerAbstract>(
            new WindowExprBigramConcatLayer(dynet_model, unit_embedding_dim, window_sz)
            );
    }
    else if(name == "attention1")
    {
        return std::shared_ptr<WindowExprProcessingLayerAbstract>(
            new WindowExprAttention1Layer(dynet_model, unit_embedding_dim, window_sz)
            );
    }
    else
    { 
        throw std::invalid_argument("unsupported window expression processing method: '" + processing_method + "'\n"
            "supporting list: concat, avg, bigram, bigram-concat, attention1\n");
    }
}

} // end of namespace experiment
} // end of namespace nn_module
} // end of namespace segmenter
} // end of namespace slnn
