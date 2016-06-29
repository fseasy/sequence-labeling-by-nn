#ifndef SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_LEXICON_FEATURE_H_

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>

#include "utils/typedeclaration.h"
#include "utils/utf8processing.hpp"
namespace slnn{

struct LexiconFeatureData
{
    unsigned char startHereFeatureIndex;
    unsigned char passHereFeatureIndex;
    unsigned char endHereFeatureIndex;
    // startHereFeatureIndex = 0 for word with 1 character
    // passHereFeatureIndex = 0 for word with less than 3 character
    // endHereFeatureIndex = 0 for word with 1 character
    LexiconFeatureData() : startHereFeatureIndex(0), passHereFeatureIndex(0), endHereFeatureIndex(0){} 
    void setStartHereFeatureIndex(unsigned char val){ startHereFeatureIndex = std::max(startHereFeatureIndex, val); }
    void setPassHereFeatureIndex(unsigned char val){ passHereFeatureIndex = std::max(passHereFeatureIndex, val); }
    void setEndHereFeatureIndex(unsigned char val){ endHereFeatureIndex = std::max(endHereFeatureIndex, val); }
    unsigned char getStartHereFeatureIndex(){ return startHereFeatureIndex; }
    unsigned char getPassHereFeatureIndex(){ return passHereFeatureIndex; }
    unsigned char getEndHereFeatureIndex(){ return endHereFeatureIndex; }
};

using LexiconFeatureDataSeq = std::vector<LexiconFeatureData>;

class LexiconFeature
{
    friend class boost::serialization::access;
public :
    // variable (functtion)
    static constexpr unsigned WordLenLimit(){ return 5u;  }
    static constexpr unsigned LexiconStartHereDictSize(){ return WordLenLimit(); }
    static constexpr unsigned LexiconEndHereDictSize(){ return WordLenLimit(); }
    static constexpr unsigned LexiconPassHereDictSize(){ return WordLenLimit() - 1u; } // 2,3,4,5

public :
    LexiconFeature() = default; // for load 
    LexiconFeature(unsigned feature_start_here_dim, unsigned feature_end_here_dim, unsigned feature_pass_here_dim);
    LexiconFeature(unsigned featureDim) : LexiconFeature(featureDim, featureDim, featureDim){};

    void set_dim(unsigned feature_start_here_dim, unsigned feature_end_here_dim, unsigned feature_pass_here_dim);
    unsigned get_feature_start_here_dim(){ return feature_start_here_dim; }
    unsigned get_feature_pass_here_dim(){ return feature_pass_here_dim; }
    unsigned get_feature_end_here_dim(){ return feature_end_here_dim; }
    unsigned get_feature_dim(){ return feature_start_here_dim + feature_pass_here_dim + feature_end_here_dim; }

    void count_word_freqency(const Seq &word_seq);
    void build_lexicon();
    void extract(const Seq &char_seq, LexiconFeatureDataSeq &lexicon_feature_seq);

    std::string get_lexicon_info();

    template<typename Archive>
    void serialize(Archive &ar, unsigned version);

private:
    unsigned feature_start_here_dim;
    unsigned feature_end_here_dim;
    unsigned feature_pass_here_dim;
    unsigned lexicon_word_max_len;
    unsigned freq_threshold; // >= freq_threshold can be add to lexicon
    std::unordered_map<std::string, unsigned> word_count_dict;
    std::unordered_set<std::string> lexicon;
};

LexiconFeature::LexiconFeature(unsigned feature_start_here_dim, unsigned feature_end_here_dim, unsigned feature_pass_here_dim)
    :feature_start_here_dim(feature_start_here_dim),
    feature_end_here_dim(feature_end_here_dim),
    feature_pass_here_dim(feature_pass_here_dim),
    lexicon_word_max_len(0)
{}

void LexiconFeature::set_dim(unsigned feature_start_here_dim, unsigned feature_end_here_dim, unsigned feature_pass_here_dim)
{
    this->feature_start_here_dim = feature_start_here_dim;
    this->feature_end_here_dim = feature_end_here_dim;
    this->feature_pass_here_dim = feature_pass_here_dim;
}

void LexiconFeature::count_word_freqency(const Seq &word_seq)
{
    for( const std::string &word : word_seq )
    {
        ++word_count_dict[word];
    }
}

void LexiconFeature::build_lexicon()
{
    // LTP version
    std::vector<unsigned> freq_list(word_count_dict.size());
    // get frequency threshold
    unsigned idx = 0;
    unsigned long long total_freq = 0;
    for( std::unordered_map<std::string, unsigned>::const_iterator iter = word_count_dict.cbegin(); iter != word_count_dict.cend(); ++iter )
    {
        freq_list[idx++] = iter->second;
        total_freq += iter->second;
    }
    sort(freq_list.begin(), freq_list.end(), std::greater<unsigned>());
    unsigned long long trunk_total_freq = total_freq * 0.9f;
    unsigned long long current_freq = 0;
    freq_threshold = 0;
    for( size_t i = 0; i < freq_list.size(); ++i )
    {
        current_freq += freq_list[i];
        if( current_freq >= trunk_total_freq ){ freq_threshold = freq_list[i]; break; }
    }
    // move word with at least 2 character and freqency greater than freq_threshold to lexicon
    lexicon_word_max_len = 0;
    for( auto iter = word_count_dict.cbegin(); iter != word_count_dict.cend(); ++iter )
    {
        if( iter->second >= freq_threshold )
        {
            size_t nr_utf8_char = UTF8Processing::utf8_char_len(iter->first);
            if( nr_utf8_char > 1)
            {
                // at least 2 word
                lexicon.insert(std::move(iter->first));
                lexicon_word_max_len = std::max(lexicon_word_max_len, nr_utf8_char);
            }
        }
    }
    // clear word count dict 
    word_count_dict.clear();
}

void LexiconFeature::extract(const Seq &char_seq, LexiconFeatureDataSeq &lexicon_feature_seq)
{
    using std::swap;
    size_t seq_len = char_seq.size();
    std::vector<int> charLenSeq(seq_len);
    for( int i = 0; i < seq_len; ++i ){ charLenSeq[i] = char_seq[i].length(); }
    LexiconFeatureDataSeq tmp_lexicon_feature_seq(seq_len);
    // Max Match
    // this lexicon feature can be look as fusion Max Match Result
    for( size_t i = 0; i < seq_len; ++i )
    {
        size_t end_pos = std::min(seq_len, i + lexicon_word_max_len);
        std::string testWord ;
        for( size_t k = i; k < end_pos; ++k ){ testWord += char_seq[k]; }
        size_t word_len = testWord.length();
        while( end_pos > i + 1 ) // skip word with one character
        {
            if( lexicon.count(testWord) ){ break; }
            size_t erase_char_len = charLenSeq[end_pos -= 1];
            testWord.erase(word_len -= erase_char_len);
        }
        size_t char_len = std::min(end_pos - i, WordLenLimit());
        tmp_lexicon_feature_seq[i].setStartHereFeatureIndex(char_len - 1);
        tmp_lexicon_feature_seq[end_pos - 1].setEndHereFeatureIndex(char_len - 1);
        if( char_len > 2 )
        {
            for( size_t k = i+1; k < end_pos-1; ++k )
            {
                tmp_lexicon_feature_seq[k].setPassHereFeatureIndex(char_len - 2);
            }
        }
    }
    swap(lexicon_feature_seq, tmp_lexicon_feature_seq);
}

std::string LexiconFeature::get_lexicon_info()
{
    std::ostringstream oss;
    oss << "lexicon size : " << lexicon.size() << " max word length : " << lexicon_word_max_len
        << " frequence threshold : " << freq_threshold << "\n"
        << "lexicon feature dim : [" << feature_start_here_dim << ", " << feature_pass_here_dim
        << ", " << feature_end_here_dim << "]\n"
        << "total dim : " << get_feature_dim() ;
    return oss.str();
}


template<typename Archive>
void LexiconFeature::serialize(Archive &ar, unsigned version)
{
    ar & feature_start_here_dim & feature_end_here_dim
        & feature_pass_here_dim & lexicon_word_max_len
        & freq_threshold;
    ar & lexicon;
}

} // end of namespace slnn
#endif