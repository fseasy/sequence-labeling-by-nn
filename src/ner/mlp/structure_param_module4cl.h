#ifndef SLNN_NER_MLP_CL_MODEL_H
#define SLNN_NER_MLP_CL_MODEL_H

#include <vector>
#include <string>

namespace slnn{
namespace ner{

struct CLModel
{

    // strucutre param
    unsigned word_dict_sz;
    unsigned postag_dict_sz;
    unsigned ner_tag_dict_sz;
    unsigned word_embed_dim;
    unsigned postag_embed_dim;

    unsigned mlp_window_sz;
    std::vector<unsigned> mlp_hidden_dim_list;
    std::string nonlinear_func_name;

    // 
};




}
} // end of namespace slnn


#endif