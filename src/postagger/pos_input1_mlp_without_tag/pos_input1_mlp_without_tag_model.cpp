#include "pos_input1_mlp_without_tag_model.h"

namespace slnn{

Input1MLPWithoutTagModel::Input1MLPWithoutTagModel()
    :Input1MLPModel(),
    input_layer(nullptr),
    mlp_hidden_layer(nullptr),
    output_layer(nullptr),
    pos_feature_layer(nullptr),
    pos_context_feature_layer(nullptr)
{}

Input1MLPWithoutTagModel:: ~Input1MLPWithoutTagModel()
{
    delete input_layer;
    delete mlp_hidden_layer;
    delete output_layer;
    delete pos_feature_layer;
    delete pos_context_feature_layer;
}

void Input1MLPWithoutTagModel::set_model_param_from_outer(const boost::program_options::variables_map &var_map)
{
    Input1MLPModel::set_model_param_from_outer(var_map);
}

void Input1MLPWithoutTagModel::build_model_structure()
{
    m = new dynet::Model();
    input_layer = new BareInput1(m, word_dict_size, word_embedding_dim, 2);
    mlp_hidden_layer = new MLPHiddenLayer(m, input_dim, mlp_hidden_dim_list, dropout_rate);
    output_layer = new SoftmaxLayer(m, mlp_hidden_dim_list.at(mlp_hidden_dim_list.size() - 1), output_dim);
    pos_feature_layer = new POSFeatureLayer(m, pos_feature);
    pos_context_feature_layer = new ContextFeatureLayer(m, input_layer->get_lookup_param());
}

void Input1MLPWithoutTagModel::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Input1 MLP Without Tag -----------------\n"
        << "vocabulary size : " << word_dict_size << " with dimension : " << word_embedding_dim << "\n"
        << "input dim : " << input_dim << "\n"
        << "mlp hidden dims : " << get_mlp_hidden_layer_dim_info() << "\n"
        << "output dim : " << output_dim << "\n"
        << "feature info : \n"
        << pos_feature.get_feature_info() << "\n"
        << "context info : \n"
        << context_feature.get_feature_info();
}


dynet::expr::Expression  
Input1MLPWithoutTagModel::build_loss(dynet::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const ContextFeatureDataSeq &context_feature_gp_seq,
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    const IndexSeq &gold_seq)
{
    pos_feature_layer->new_graph(cg);
    pos_context_feature_layer->new_graph(cg);
    input_layer->new_graph(cg);
    mlp_hidden_layer->new_graph(cg);
    output_layer->new_graph(cg);
    unsigned sent_len = input_seq.size();

    std::vector<dynet::expr::Expression> input_exprs(sent_len);
    std::vector<dynet::expr::Expression> tmp_feature_cont(2) ;
    for( unsigned i = 0 ; i < sent_len; ++i )
    {
        tmp_feature_cont.at(0) = pos_context_feature_layer->build_feature_expr(
            context_feature_gp_seq.at(i)
            ) ;
        tmp_feature_cont.at(1) = pos_feature_layer->build_feature_expr(features_gp_seq.at(i));
        input_exprs.at(i) = input_layer->build_input(input_seq.at(i), tmp_feature_cont);
    }
    std::vector<dynet::expr::Expression> output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, output_exprs);
    return output_layer->build_output_loss(output_exprs, gold_seq);
}

void 
Input1MLPWithoutTagModel::predict(dynet::ComputationGraph &cg,
    const IndexSeq &input_seq,
    const ContextFeatureDataSeq &context_feature_gp_seq,
    const POSFeature::POSFeatureIndexGroupSeq &features_gp_seq,
    IndexSeq &pred_seq)
{
    pos_feature_layer->new_graph(cg);
    pos_context_feature_layer->new_graph(cg);
    input_layer->new_graph(cg);
    mlp_hidden_layer->new_graph(cg);
    output_layer->new_graph(cg);
    unsigned sent_len = input_seq.size();

    std::vector<dynet::expr::Expression> input_exprs(sent_len);
    std::vector<dynet::expr::Expression> tmp_feature_cont(2) ;
    for( unsigned i = 0 ; i < sent_len; ++i )
    {
        tmp_feature_cont.at(0) = pos_context_feature_layer->build_feature_expr(
            context_feature_gp_seq.at(i)
            ) ;
        tmp_feature_cont.at(1) = pos_feature_layer->build_feature_expr(features_gp_seq.at(i));
        input_exprs.at(i) = input_layer->build_input(input_seq.at(i), tmp_feature_cont);
    }
    std::vector<dynet::expr::Expression> output_exprs;
    mlp_hidden_layer->build_graph(input_exprs, output_exprs);
    output_layer->build_output(output_exprs, pred_seq);
}


} // end of namespace slnn