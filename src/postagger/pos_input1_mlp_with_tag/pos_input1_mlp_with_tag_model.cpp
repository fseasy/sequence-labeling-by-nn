#include "pos_input1_mlp_with_tag_model.h"

namespace slnn{

POSInput1MLPWithTagModel::POSInput1MLPWithTagModel()
    :Input1MLPModel(),
    word_expr_layer(nullptr),
    tag_expr_layer(nullptr),
    pos_feature_layer(nullptr),
    pos_context_feature_layer(nullptr),
    mlp_hidden_layer(nullptr),
    output_layer(nullptr)
{}

POSInput1MLPWithTagModel:: ~POSInput1MLPWithTagModel()
{
    delete word_expr_layer;
    delete tag_expr_layer;
    delete pos_feature_layer;
    delete pos_context_feature_layer;
    delete mlp_hidden_layer;
    delete output_layer;
}

void POSInput1MLPWithTagModel::set_model_param(const boost::program_options::variables_map &var_map)
{
    nonlinear_func_indicate = var_map["mlp_nonlinear_func"].as<std::string>();
    tag_embedding_dim = var_map["tag_embedding_dim"].as<unsigned>();
    Input1MLPModel::set_model_param(var_map);
    input_dim += tag_embedding_dim; // update input dim
}

void POSInput1MLPWithTagModel::build_model_structure()
{
    if( nonlinear_func_indicate == "rectify" ){ nonlinear_func = &cnn::expr::rectify;  }
    else if( nonlinear_func_indicate == "tanh" ){ nonlinear_func = &cnn::expr::tanh;  }
    else { throw std::runtime_error("unsupported nonlinear function : " + nonlinear_func_indicate) ;  }
    m = new cnn::Model();
    word_expr_layer = new Index2ExprLayer(m, word_dict_size, word_embedding_dim);
    tag_expr_layer = new ShiftedIndex2ExprLayer(m, output_dim, tag_embedding_dim, ShiftedIndex2ExprLayer::RightShift, 1);
    pos_feature_layer = new POSFeatureLayer(m, pos_feature);
    pos_context_feature_layer = new ContextFeatureLayer<POSContextFeature::ContextSize>(m, word_expr_layer->get_lookup_param());
    mlp_hidden_layer = new MLPHiddenLayer(m, input_dim, mlp_hidden_dim_list, dropout_rate, nonlinear_func);
    output_layer = new SoftmaxLayer(m, mlp_hidden_dim_list.back(), output_dim);
}

void POSInput1MLPWithTagModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Input1 MLP With Tag -----------------\n"
        << "word vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "tag number : " << output_dim << " with dimension : " << tag_embedding_dim << "\n"
        << "input dim : " << input_dim << "\n"
        << "mlp hidden dims : " << get_mlp_hidden_layer_dim_info() << "\n"
        << "output dim : " << output_dim << "\n"
        << "feature info : \n"
        << pos_feature.get_feature_info() << "\n"
        << "context info : \n"
        << context_feature.get_context_info();
}


cnn::expr::Expression  
POSInput1MLPWithTagModel::build_loss(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    const IndexSeq &gold_seq)
{
    word_expr_layer->new_graph(cg);
    tag_expr_layer->new_graph(cg);
    pos_feature_layer->new_graph(cg);
    pos_context_feature_layer->new_graph(cg);
    mlp_hidden_layer->new_graph(cg);
    output_layer->new_graph(cg);
    unsigned sent_len = input_seq.size();

    std::vector<cnn::expr::Expression> word_exprs(sent_len),
        pre_tag_exprs(sent_len),
        pos_feature_exprs(sent_len),
        context_feature_exprs(sent_len);
    word_expr_layer->index_seq2expr_seq(input_seq, word_exprs);
    tag_expr_layer->index_seq2expr_seq(gold_seq, pre_tag_exprs); // shifted 
    pos_feature_layer->build_feature_exprs(features_gp_seq, pos_feature_exprs);
    pos_context_feature_layer->build_feature_exprs(context_feature_gp_seq, context_feature_exprs);
    // concate
    std::vector<cnn::expr::Expression> input_exprs(sent_len);
    StaticConcatenateLayer::concatenate_exprs(std::vector<std::vector<cnn::expr::Expression> *>({
        &word_exprs, &pre_tag_exprs, &pos_feature_exprs, &context_feature_exprs }), input_exprs);

    std::vector<cnn::expr::Expression> output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, output_exprs);
    return output_layer->build_output_loss(output_exprs, gold_seq);
}

void 
POSInput1MLPWithTagModel::predict(cnn::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq,
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    IndexSeq &pred_seq)
{
    using std::swap;
    word_expr_layer->new_graph(cg);
    tag_expr_layer->new_graph(cg);
    pos_feature_layer->new_graph(cg);
    pos_context_feature_layer->new_graph(cg);
    mlp_hidden_layer->new_graph(cg);
    output_layer->new_graph(cg);
    unsigned sent_len = input_seq.size();

    std::vector<Index> tmp_pred_seq(sent_len);
    cnn::expr::Expression pre_tag_expr = tag_expr_layer->get_padding_expr(0);
    for( int i = 0; i < sent_len ; ++i )
    {
        cnn::expr::Expression word_expr = word_expr_layer->index2expr(input_seq[i]),
            pos_feature_expr = pos_feature_layer->build_feature_expr(features_gp_seq[i]),
            context_feature_expr = pos_context_feature_layer->build_feature_expr(context_feature_gp_seq[i]);
        cnn::expr::Expression input_expr = StaticConcatenateLayer::concatenate_exprs(std::vector<cnn::expr::Expression>({
            word_expr, pre_tag_expr, pos_feature_expr, context_feature_expr
        }));
        cnn::expr::Expression hidden_expr = mlp_hidden_layer->build_graph(input_expr);
        tmp_pred_seq[i] = output_layer->build_output(hidden_expr);
        pre_tag_expr = tag_expr_layer->index2expr(tmp_pred_seq[i]);
    }
    swap(pred_seq, tmp_pred_seq);
}


} // end of namespace slnn