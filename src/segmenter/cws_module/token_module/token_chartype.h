#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_TOKEN_CHARTYPE_H_
#define SLNN_SEGMNETER_CWS_MODULE_TOKEN_MODULE_TOKEN_CHARTYPE_H_
#include <unordered_set>
#include <vector>
#include <memory>
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmenter{
namespace token_module{

class TokenChartype
{
public:
    TokenChartype(){}
public:
    static constexpr unsigned size() { return 4U; }
    static constexpr Index DefaultType = 0;
    static constexpr Index DigitType = 1;
    static constexpr Index PuncType = 2;
    static constexpr Index LetterType = 3;
public:
    static std::shared_ptr<std::vector<Index>> extract(const std::u32string& charseq);
private:
    static const std::unordered_set<char32_t> DigitTypeCharDict;
    static const std::unordered_set<char32_t> PuncTypeCharDict;
    static const std::unordered_set<char32_t> LetterTypeCharDict;
public:
    static bool isDigit(char32_t uc) { return DigitTypeCharDict.count(uc) > 0; }
    static bool isPunc(char32_t uc) { return PuncTypeCharDict.count(uc) > 0; }
    static bool isLetter(char32_t uc) { return LetterTypeCharDict.count(uc) > 0; }
};

inline 
std::shared_ptr<std::vector<Index>> TokenChartype::extract(const std::u32string& charseq)
{
    std::shared_ptr<std::vector<Index>> type_feat(new std::vector<Index>(charseq.size()));
    
    for(unsigned i = 0; i < charseq.size(); ++i )
    {
        auto uc = charseq[i];
        if( isPunc(uc) ){ type_feat->at(i) = PuncType; }
        else if( isDigit(uc) ){ type_feat->at(i) = DigitType; }
        else if( isLetter(uc) ){ type_feat->at(i) = LetterType; }
        else{ type_feat->at(i) = DefaultType; }
    }
    return type_feat;
}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif