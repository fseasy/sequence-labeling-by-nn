#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#define SLNN_SEGMNETER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#include <unordered_set>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace token_module{

class TokenLexicon
{
    friend class boost::serialization::access;
public:
    TokenLexicon(unsigned maxlen4feature = 5);
public:
    void set_maxlen4feature(unsigned maxlen4feature){ this->maxlen4feature = maxlen4feature; }
    void build_inner_lexicon_from_training_data(std::ifstream &training_is);
    std::shared_ptr<std::vector<std::vector<Index>>> extract(const std::u32string& charseq) const;
    std::size_t size() const { return maxlen4feature; }
    std::string get_lexicon_feature_info() const;
private:
    template <class Archive>
    void save(Archive &ar, const unsigned int) const;
    template <class Archive>
    void load(Archive &ar, const unsigned int);
    BOOST_SERIALIZATION_SPLIT_MEMBER();
private:
    unsigned maxlen4feature; // pre-define.
    unsigned word_maxlen_in_lexicon; // calc according to final lexicon
    std::unordered_set<std::u32string> inner_lexicon; // can't serialize directly
    unsigned freq_threshold; // the frequent threshold to dethermine where an token is a word.  (For DEBUG)
};

/**********************
 * Inline/template Implementation
 **********************/

inline
std::string TokenLexicon::get_lexicon_feature_info() const
{
    std::ostringstream oss;
    oss << "+ Lexicon Feature info:"
        << "| feature max len(" << maxlen4feature << ")"
        << " word max len in lexicon(" << word_maxlen_in_lexicon << ")\n"
        << "| lexicon size(" << inner_lexicon.size() << ") frequent threshold(" << freq_threshold << ")\n"
        << "== - - - - -";
    return oss.str();
}

template <class Archive>
void TokenLexicon::save(Archive &ar, const unsigned int) const
{
    ar &maxlen4feature &word_maxlen_in_lexicon;
    unsigned lexicon_sz = inner_lexicon.size();
    ar &lexicon_sz;
    for( const std::u32string& word : inner_lexicon )
    {
        std::vector<unsigned> unicode_pnt_list(word.begin(), word.end());
        ar &unicode_pnt_list;
    }
}

template <class Archive>
void TokenLexicon::load(Archive &ar, const unsigned int)
{
    ar &maxlen4feature &word_maxlen_in_lexicon;
    unsigned lexicon_sz;
    ar &lexicon_sz;
    for( unsigned i = 0; i < lexicon_sz; ++i )
    {
        std::vector<unsigned> unicode_pnt_list;
        ar &unicode_pnt_list;
        inner_lexicon.emplace(unicode_pnt_list.begin(), unicode_pnt_list.end());
    }
}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif