#include "nn_cws_specific_output_layer.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "segmenter/cws_module/token_module/cws_tag_utility.h"
namespace slnn{
namespace segmenter{
namespace nn_module{
namespace experiment{

/***************************************************
* Segmentor output :  Simple Bare output (re-write @2016-10-13)
***************************************************/

SegmenterClassificationBareOutput::
SegmenterClassificationBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim)
    :SimpleBareOutput(m, input_dim, output_dim)
{}

void SegmenterClassificationBareOutput::
build_output(const std::vector<dynet::expr::Expression> &input_expr_seq, std::vector<Index> &out_pred_seq)
{
    using std::swap;
    std::size_t len = input_expr_seq.size();
    if( 1 == len ) // Special Condition : len = 1
    {
        out_pred_seq = { segmenter::Tag::TAG_S_ID };
        return ;
    }
    std::vector<Index> tmp_pred_out(len);
    Index pre_tag_id = segmenter::Tag::TAG_NONE_ID;
    for (std::size_t i = 0; i < len - 1; ++i) // select first and middle tags
    {
        dynet::expr::Expression out_expr = softmax_layer.build_graph(input_expr_seq[i]) ;
        std::vector<dynet::real> out_probs = dynet::as_vector(pcg->get_value(out_expr));
        Index max_prob_tag_in_constrain = segmenter::token_module::select_best_tag_constrained(out_probs , i , pre_tag_id );
        tmp_pred_out[i] = max_prob_tag_in_constrain ;
        pre_tag_id = max_prob_tag_in_constrain ;
    }
    if( pre_tag_id == segmenter::Tag::TAG_M_ID||  // select last tag
        pre_tag_id == segmenter::Tag::TAG_B_ID)
    { 
        tmp_pred_out[len - 1] = segmenter::Tag::TAG_E_ID; 
    }
    else { tmp_pred_out[len - 1] = segmenter::Tag::TAG_S_ID; }
    swap(out_pred_seq, tmp_pred_out);
}

std::shared_ptr<BareOutputBase>
create_segmenter_output_layer(const std::string& layer_type, dynet::Model *dynet_model, unsigned input_dim, unsigned output_dim)
{
    std::string name(layer_type);
    for( char &c : name ){ c = ::tolower(c); }
    if( name == "classification" || name == "cl" )
    {
        // just classification by the current state. No tag constraint of CWS.
        return std::shared_ptr<BareOutputBase>(new SimpleBareOutput(dynet_model, input_dim, output_dim));
    }
    else if( name == "pretag" )
    {
        throw std::logic_error("havn't implemented.");
    }
    else if( name == "crf" )
    {
        throw std::logic_error("havn't implemented.");
    }
    else if( name == "classification_limit" || name == "cl_limit" )
    {
        // the original classification decoding method.
        // decode the current tag Y with the condition of previous tag YP and using the hard limit: YP -> Y is the valid transition. 
        return std::shared_ptr<BareOutputBase>(new SegmenterClassificationBareOutput(dynet_model, input_dim, output_dim));
    }
    else
    {
        throw std::invalid_argument("un-spport output layer type: '" + layer_type + "'\n"
            "supported including cl, pretag, crf.\n");
    }
}

} // end of namespace experiment
} // end of namespace nn-module
} // end of namespace segmeter
} // end of namespace slnn