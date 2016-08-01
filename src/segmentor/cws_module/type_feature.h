#ifndef SLNN_SEGMENTOR_CWS_MODULE_TYPE_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_TYPE_FEATURE_H_
#include <unordered_set>
#include <string>
#include <boost/serialization/access.hpp>
#include "utils/typedeclaration.h"
namespace slnn{

namespace slnn_char_type{

class Utf8CharTypeDict
{
private:
    static const std::unordered_set<std::string> DigitTypeCharDict;
    static const std::unordered_set<std::string> PuncTypeCharDict;
    static const std::unordered_set<std::string> LetterTypeCharDict;
public:
    bool isDigit(const std::string& u8char) const { return DigitTypeCharDict.count(u8char); }
    bool isPunc(const std::string& u8char) const { return PuncTypeCharDict.count(u8char); }
    bool isLetter(const std::string& u8char) const { return LetterTypeCharDict.count(u8char); }
};

} // end of namespcae slnn_char_type 

inline
slnn_char_type::Utf8CharTypeDict& getCharTypeDict()
{
    static slnn_char_type::Utf8CharTypeDict chartypeDict;
    return chartypeDict;
}

using CharTypeFeatureData = Index;
using CharTypeFeatureDataSeq = IndexSeq;

class CharTypeFeature
{
    friend class boost::serialization::access;
public :
    static constexpr Index DefaultType(){ return 0 ; } // According to Effective C++ , Item 4. may be it is not so necessary
    static constexpr Index DigitType(){ return 1; }    // If it will not be used to initialize another 
    static constexpr Index PuncType(){ return 2; }
    static constexpr Index LetterType(){ return 3; }
    static constexpr size_t FeatureDictSize(){ return 4; }
    constexpr CharTypeFeature(unsigned feature_dim = 3) : feature_dim(feature_dim){};
    void extract(const Seq &char_seq, IndexSeq &chartype_feature_seq) const;
    unsigned get_feature_dim() const { return feature_dim; }
    void set_dim(unsigned feature_dim){ this->feature_dim = feature_dim; }

    template<typename Archive>
    void serialize(Archive &ar, unsigned);
private:
    unsigned feature_dim;
};

inline
void CharTypeFeature::extract(const Seq &char_seq, IndexSeq &chartype_feature_seq) const
{
    using std::swap;
    size_t len = char_seq.size();
    IndexSeq tmp_feature_seq(len);
    for( size_t i = 0; i < len; ++i )
    {
        const std::string &u8char = char_seq.at(i);
        if( getCharTypeDict().isDigit(u8char) ){ tmp_feature_seq[i] = DigitType(); }
        else if( getCharTypeDict().isPunc(u8char) ){ tmp_feature_seq[i] = PuncType(); }
        else if( getCharTypeDict().isLetter(u8char) ){ tmp_feature_seq[i] = LetterType(); }
        else { tmp_feature_seq[i] = DefaultType(); }
    }
    swap(chartype_feature_seq, tmp_feature_seq);
}

template<typename Archive>
void CharTypeFeature::serialize(Archive &ar, unsigned)
{
    ar & feature_dim;
}

} // end of namespace slnn

#endif