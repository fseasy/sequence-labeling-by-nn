#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include "input1_mlp_model_no_feature.h"
#include "utils/utf8processing.hpp"

using namespace std;
using namespace cnn;
namespace slnn{
const string Input1MLPModelNoFeature::UNK_STR = "unk_str";
const string Input1MLPModelNoFeature::StrOfReplaceNumber = "##";
const size_t Input1MLPModelNoFeature::LenStrOfRepalceNumber = 2;
constexpr size_t Input1MLPModelNoFeature::PostaggerContextSize ;

Input1MLPModelNoFeature::Input1MLPModelNoFeature() 
    :m(nullptr),
    word_dict_wrapper(word_dict),
    context_feature(word_dict_wrapper)
{}

Input1MLPModelNoFeature::~Input1MLPModelNoFeature()
{
    delete m;
}

void Input1MLPModelNoFeature::set_model_param(const boost::program_options::variables_map &var_map)
{
    assert(word_dict.is_frozen() && postag_dict.is_frozen() ) ;

    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    set_replace_threshold(replace_freq_threshold, replace_prob_threshold);

    word_embedding_dim = var_map["word_embedding_dim"].as<unsigned>() ;

    // parse hidden dims 
    std::vector<std::string> dim_str_cont;
    std::string hidden_dim_list_str = var_map["mlp_hidden_dim_list"].as<std::string>();
    boost::trim_if(hidden_dim_list_str, boost::is_any_of("\", ")); 
    boost::split(dim_str_cont, hidden_dim_list_str, boost::is_any_of(", ")); // split by space of comma
    mlp_hidden_dim_list.resize(dim_str_cont.size());
    try
    {
        transform(dim_str_cont.begin(), dim_str_cont.end(), mlp_hidden_dim_list.begin(),
            [](const string &num_str){ return stoul(num_str); });
    }
    catch( invalid_argument &e )
    {
        cerr << "bad argument for 'mlp_hidden_dim_list' : " << var_map["mlp_hidden_dim_list"].as<string>();
        throw e ;
    }

    dropout_rate = var_map["dropout_rate"].as<cnn::real>() ;

   
    word_dict_size = word_dict.size() ;
    output_dim = postag_dict.size() ;
    input_dim = word_embedding_dim + context_feature.calc_context_feature_dim(word_embedding_dim);
}

void Input1MLPModelNoFeature::input_seq2index_seq(const Seq &sent, 
    const Seq &postag_seq,
    IndexSeq &index_sent, 
    IndexSeq &index_postag_seq,
    POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq)
{
    using std::swap;
    assert(sent.size() == postag_seq.size());
    size_t seq_len = sent.size();
    IndexSeq tmp_sent_index_seq(seq_len),
        tmp_postag_index_seq(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        tmp_sent_index_seq[i] = word_dict_wrapper.Convert(
            UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber)
        );
        tmp_postag_index_seq[i] = postag_dict.Convert(postag_seq[i]);
    }
    ContextFeatureExtractor::extract<PostaggerContextSize>(tmp_sent_index_seq, context_feature_gp_seq);

    swap(index_sent, tmp_sent_index_seq);
    swap(index_postag_seq, tmp_postag_index_seq);
}

void Input1MLPModelNoFeature::input_seq2index_seq(const Seq &sent, 
    IndexSeq &index_sent, 
    POSContextFeature::ContextFeatureIndexGroupSeq &context_feature_gp_seq)
{
    using std::swap;
    size_t seq_len = sent.size();
    IndexSeq tmp_sent_index_seq(seq_len);
    for( size_t i = 0 ; i < seq_len; ++i )
    {
        tmp_sent_index_seq[i] = word_dict_wrapper.Convert(
            UTF8Processing::replace_number(sent[i], StrOfReplaceNumber, LenStrOfRepalceNumber)
        );
    }
    ContextFeatureExtractor::extract<PostaggerContextSize>(tmp_sent_index_seq, context_feature_gp_seq);
    
    swap(index_sent, tmp_sent_index_seq);
}

std::string Input1MLPModelNoFeature::get_mlp_hidden_layer_dim_info()
{
    ostringstream oss;
    oss << "[ " << mlp_hidden_dim_list.at(0) ;
    for(size_t i = 1 ; i < mlp_hidden_dim_list.size() ; ++i)
    {
        oss << ", " << mlp_hidden_dim_list.at(i);
    }
    oss << " ]" ;
    return oss.str();
}


} // end of namespace slnn