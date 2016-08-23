#ifndef UTILS_WORD2VEC_EMBEDDING_HELPER_H_
#define UTILS_WORD2VEC_EMBEDDING_HELPER_H_

#include <fstream>

#include "cnn/cnn.h"
#include "cnn/dict.h"

namespace slnn{
struct Word2vecEmbeddingHelper
{
    /* bulid_fixed_dict
    *
    * PARAMES
    * -------
    * is : [in] , ifstream .
    *      reference to wordembedding file stream
    * fixed_dict : [in], cnn::Dict
    *              reference to fixed dict ,
    * unk_str : [in] , string
    *           to build UNK
    * dict_size : [out] , pointer to unsigned [optional]
    *             after build the dict , set the dict size(if given)
    * embedding_dim : [out] , pointer to unsigned [optional]
    *                 after build the dict , set the embedding dim (if given)
    * RETURN
    * ------
    * void
    */
    static
        void build_fixed_dict(std::ifstream &is, cnn::Dict &fixed_dict, const std::string &unk_str,
            unsigned *p_dict_size = nullptr, unsigned *p_embedding_dim = nullptr);

    /* load_fixed_embedding
    * PARAMES
    * -------
    * is : [in] , ifstream
    *      wordembedding fstream
    * fixed_dict : [in], cnn::Dict&
    *      dict to map word 2 index
    * fixed_word_dim : [in], unsigned
           for check when loading word embedding
    * fixed_lookup_param : [in], cnn::LookupParameters*
    *      to store the word embedding
    * RETURN
    * ------
    * void
    */
    static
        void load_fixed_embedding(std::ifstream &is, cnn::Dict &fixed_dict, unsigned fixed_word_dim, cnn::LookupParameters *fixed_lookup_param);

    static float calc_hit_rate(cnn::Dict &fixed_dict, cnn::Dict &dynamic_dict, const std::string &fixed_dict_unk_str);
};

} // end of namespace slnn
#endif 