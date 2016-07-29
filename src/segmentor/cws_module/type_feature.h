#ifndef SLNN_SEGMENTOR_CWS_MODULE_TYPE_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_TYPE_FEATURE_H_
#include <unordered_set>
#include <string>
namespace slnn{

namespace slnn_char_type{

struct Utf8CharTypeDict
{
    static std::unordered_set<std::string> DigitTypeCharDict;
    static std::unordered_set<std::string> PuncTypeCharDict;
    static std::unordered_set<std::string> LetterTypeCharDict;
};

} // end of namespcae slnn_char_type 

} // end of namespace slnn

#endif