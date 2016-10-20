#include <boost/log/trivial.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "word2vec_embedding_helper.h"
#include "utils/typedeclaration.h"
using namespace std;
using namespace dynet;
namespace slnn{

void Word2vecEmbeddingHelper::build_fixed_dict(ifstream &is, Dict &fixed_dict, const string &unk_str,
    unsigned *p_dict_size, unsigned *p_embedding_dim)
{
    BOOST_LOG_TRIVIAL(info) << "initialize fixed dict .";
    std::string line;
    std::vector<std::string> split_cont;
    getline(is, line); // first line should be the infomation : word-dict-size , word-embedding-dimension
    boost::split(split_cont, line, boost::is_any_of(" "));
    unsigned fixed_dict_sz,
        fixed_word_dim;
    bool is_standard_word2vec_format ;
    if( 2U != split_cont.size() )
    {
        // not standard word2vec file . may be it's the only embedding
        is_standard_word2vec_format = false ;
        fixed_dict_sz = 0 ;
        fixed_word_dim = split_cont.size() - 1 ;
        is.clear();
        is.seekg(0, is.beg) ;
    }
    else
    {
        is_standard_word2vec_format = true ;
        fixed_dict_sz = std::stol(split_cont[0]) + 1; // another UNK
        fixed_word_dim = std::stol(split_cont[1]);
    }
    // read all words and add to dc_m.fixed_dict
    while( getline(is, line) )
    {
        std::string::size_type delim_pos = line.find(" ");
        assert(delim_pos != std::string::npos);
        std::string word = line.substr(0, delim_pos);
        fixed_dict.convert(word);  // add to dict
    }
    //  freeze & add unk to fixed_dict
    fixed_dict.freeze();
    fixed_dict.set_unk(unk_str);
    if( p_dict_size )
    {
        if( is_standard_word2vec_format ){ assert(fixed_dict_sz == fixed_dict.size()) ; }
        *p_dict_size = fixed_dict.size() ;
    }
    if( p_embedding_dim ){ *p_embedding_dim = fixed_word_dim; }

    BOOST_LOG_TRIVIAL(info) << "build fixed dict done .";
}

void Word2vecEmbeddingHelper::load_fixed_embedding(std::ifstream &is, dynet::Dict &fixed_dict, unsigned fixed_word_dim, dynet::LookupParameter fixed_lookup_param)
{
    // set lookup parameters from outer word embedding
    // using words_loopup_param.initialize( word_id , value_vector )
    BOOST_LOG_TRIVIAL(info) << "load pre-trained word embedding .";
    std::string line;
    std::vector<std::string> split_cont;
    getline(is, line); // first line is the infomation , skip
    unsigned long line_cnt = 0; // for warning when read embedding error
    std::vector<dynet::real> embedding_vec(fixed_word_dim, 0.f);
    while( getline(is, line) )
    {
        ++line_cnt;
        boost::trim_right(line);
        boost::split(split_cont, line, boost::is_any_of(" "));
        if( fixed_word_dim + 1 != split_cont.size() )
        {
            BOOST_LOG_TRIVIAL(warning) << "bad word dimension : `" << split_cont.size() - 1 << "` at line " << line_cnt;
            continue;
        }
        std::string &word = split_cont.at(0);
        Index word_id = fixed_dict.convert(word);
        for( size_t idx = 1; idx < split_cont.size(); ++idx )
        {
            embedding_vec[idx - 1] = std::stof(split_cont[idx]);
        }
        fixed_lookup_param.initialize(word_id, embedding_vec);
    }
    BOOST_LOG_TRIVIAL(info) << "load fixed embedding done ." ;
}

float Word2vecEmbeddingHelper::calc_hit_rate(dynet::Dict &fixed_dict, dynet::Dict &dynamic_dict, const std::string &fixed_dict_unk_str)
{
    unsigned fixed_dict_sz = fixed_dict.size(),
        dynamic_dict_sz = dynamic_dict.size() - 1; // except the fdynamic dict unk str
    unsigned nr_hit_word = 0 ;
    Index fixed_unk = fixed_dict.convert(fixed_dict_unk_str);
    for( unsigned word_key = 0; word_key < fixed_dict_sz; ++word_key )
    {
        Index word_idx = word_key; // in fact, to avoid compare between signed and unsigned
        if( word_idx == fixed_unk ){ continue; }
        string word = fixed_dict.convert(word_idx);
        if( dynamic_dict.Contains(word) ){ ++nr_hit_word; }
    }
    float hit_rate = (dynamic_dict_sz ? static_cast<float>(nr_hit_word) / dynamic_dict_sz : 0.f) * 100 ;
    BOOST_LOG_TRIVIAL(info) << "intersected words num : " << nr_hit_word << "\t"
        << "dynamic word num : " << dynamic_dict_sz << "\t"
        << "fixed word num : " << fixed_dict_sz - 1 << "\n"
        << "hit rate = nr_intersected_word / nr_dynamic_word = " << hit_rate << " %" ;
    return hit_rate;
}

} // end of namespace slnn
