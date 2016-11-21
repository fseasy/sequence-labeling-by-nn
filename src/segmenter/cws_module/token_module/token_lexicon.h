#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#define SLNN_SEGMNETER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#include <unordered_set>
#include <string>
namespace slnn{
namespace segmenter{
namespace token_module{

class TokenLexicon
{
public:
    TokenLexicon(unsigned maxlen4feature = 5):lexicon_len_upper_bound(maxlen4feature){};
public:

private:
    unsigned lexicon_len_upper_bound; // pre-define.
    unsigned word_maxlen_in_lexicon; // calc according to final lexicon
    std::unordered_set<std::u32string> inner_lexicon; // can't serialize directly
};



} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif