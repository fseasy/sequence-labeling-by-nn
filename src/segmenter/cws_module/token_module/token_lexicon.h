#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#define SLNN_SEGMNETER_CWS_MODULE_TOKEN_MODULE_TOKEN_LEXICON_H_
#include <unordered_set>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace token_module{

class TokenLexicon
{
public:
    TokenLexicon(unsigned maxlen4feature = 5);
public:
    void build_inner_lexicon_from_training_data(std::ifstream &training_is);
    std::shared_ptr<std::vector<std::vector<Index>>> extract(const std::u32string& charseq) const;
private:
    unsigned maxlen4feature; // pre-define.
    unsigned word_maxlen_in_lexicon; // calc according to final lexicon
    std::unordered_set<std::u32string> inner_lexicon; // can't serialize directly
};



} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif