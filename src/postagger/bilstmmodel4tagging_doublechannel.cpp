#include <fstream>
#include <vector>
#inlcude <string>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "bilstmmodel4tagging_doublechannel.h"

using namespace std;

namespace slnn{

DoubleChannelModel4POSTAG::DoubleChannelModel4POSTAG()
    : m(nullptr),
    merge_doublechannel_layer(nullptr) ,
    bilstm_layer(nullptr),
    merge_bilstm_and_pretag_layer(nullptr),
    tag_output_linear_layer(nullptr),
    dynamic_dict_wrapper(dynamic_dict)
{}

DoubleChannelModel4POSTAG::~DoubleChannelModel4POSTAG()
{
    if (m) delete m;
    if (merge_doublechannel_layer) delete merge_doublechannel_layer;
    if (bilstm_layer) delete bilstm_layer;
    if (merge_bilstm_and_pretag_layer) delete merge_bilstm_and_pretag_layer;
    if (tag_output_linear_layer) delete tag_output_linear_layer;
}

void DoubleChannelModel4POSTAG::freeze_dict_and_add_UNK()
{
    if (dynamic_dict.is_frozen() || fixed_dict.is_frozen() || postag_dict.is_frozen()) return;
    dynamic_dict.Freeze();
    fixed_dict.Freeze();
    postag_dict.Freeze();

    dynamic_dict.SetUnk(UNK_STR);
    fixed_dict.SetUnk(UNK_STR);
}

void DoubleChannelModel4POSTAG::set_partial_model_structure_param_from_outer(boost::program_options::variables_map &varmap)
{
    dynamic_embedding_dim = varmap["dynamic_embedding_dim"].as<unsigned>();
    postag_embedding_dim = varmap["postag_embedding_dim"].as<unsigned>();
    nr_lstm_stacked_layer = varmap["nr_lstm_stacked_layer"].as<unsigned>();
    lstm_x_dim = varmap["lstm_x_dim"].as<unsigned>();
    lstm_h_dim = varmap["lstm_h_dim"].as<unsigned>();
    tag_layer_hidden_dim = varmap["tag_layer_hidden_dim"].as<unsigned>();

    string word2vec_embedding_path = varmap["word2vec_embedding_path"].as<string>();
    ifstream fis(word2vec_embedding_path); // do not check the open status , should be check outer
    string headerline;
    getline(fis , headerline);
    boost::trim_right(headerline);
    vector<string> split_cont;
    boost::split(split_cont, headerline, boost::is_any_of(" "));
    assert(2U == split_cont.size());
    fixed_embedding_dict_size = stol(split_cont[0]);
    fixed_embedding_dim = stol(split_cont[1]);
    // 8  parameters has been inited
}



void DoubleChannelModel4POSTAG::set_partial_model_structure_param_from_inner()
{
    dynamic_embedding_dict_size = dynamic_dict.size();
    tag_layer_output_dim = postag_dict.size();
    assert(fixed_dict.size() == fixed_embedding_dict_size);
}
} // end of namespace
