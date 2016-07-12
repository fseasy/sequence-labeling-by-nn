#include "lexicon_feature.h"

namespace slnn{

LexiconFeature::LexiconFeature(unsigned start_here_feature_dim, unsigned pass_here_feature_dim, unsigned end_here_feature_dim)
    :start_here_feature_dim(start_here_feature_dim),
    pass_here_feature_dim(pass_here_feature_dim),
    end_here_feature_dim(end_here_feature_dim)
{}

void LexiconFeature::set_dim(unsigned start_here_feature_dim, unsigned pass_here_feature_dim, unsigned end_here_feature_dim)
{
    this->start_here_feature_dim = start_here_feature_dim;
    this->pass_here_feature_dim = pass_here_feature_dim;
    this->end_here_feature_dim = end_here_feature_dim;
}

void LexiconFeature::count_word_frequency(const Seq &word_seq)
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
    std::sort(freq_list.begin(), freq_list.end(), std::greater<unsigned>());
    unsigned long long trunk_total_freq = total_freq * 0.9L;
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
            unsigned nr_utf8_char = UTF8Processing::utf8_char_len(iter->first);
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

void LexiconFeature::extract(const Seq &char_seq, LexiconFeatureDataSeq &lexicon_feature_seq) const 
{
    using std::swap;
    // use unsigned and unsigned char , instead of size_t 
    unsigned seq_len = char_seq.size();
    std::vector<unsigned char> u8char_byte_size_seq(seq_len);
    for( unsigned i = 0; i < seq_len; ++i ){ u8char_byte_size_seq[i] = char_seq[i].length(); }
    LexiconFeatureDataSeq tmp_lexicon_feature_seq(seq_len);
    // Max Match
    // this lexicon feature can be look as fusion Max Match Result
    for( unsigned i = 0; i < seq_len; ++i ) // index increasing one by one , instead of doing like MM which skip word 
    {
        unsigned end_pos = std::min(seq_len, i + lexicon_word_max_len);
        std::string testWord ;
        for( unsigned k = i; k < end_pos; ++k ){ testWord += char_seq[k]; }
        unsigned word_byte_size = testWord.length();
        while( end_pos > i + 1 ) // skip word with one character
        {
            if( lexicon.count(testWord) > 0 ){ break; }
            unsigned erase_char_byte_size = u8char_byte_size_seq[end_pos -= 1];
            testWord.erase(word_byte_size -= erase_char_byte_size);
        }
        unsigned char char_len = static_cast<unsigned char>( std::min(end_pos - i, WordLenLimit()) );
        tmp_lexicon_feature_seq[i].set_start_here_feature(char_len);
        for( unsigned k = i+1; k < end_pos-1; ++k )
        {
            tmp_lexicon_feature_seq[k].set_pass_here_feature(char_len);
        }
        tmp_lexicon_feature_seq[end_pos - 1].set_end_here_feature(char_len);
    }
    swap(lexicon_feature_seq, tmp_lexicon_feature_seq);
}

std::string LexiconFeature::get_lexicon_info() const
{
    std::ostringstream oss;
    oss << "lexicon size : " << lexicon.size() << " max word length : " << lexicon_word_max_len
        << " frequence threshold : " << freq_threshold << "\n"
        << "lexicon feature dim : [" << get_start_here_feature_dim() << ", " << get_pass_here_feature_dim()
        << ", " << get_end_here_feature_dim() << "]\n"
        << "total dim : " << get_feature_dim() ;
    return oss.str();
}


} // end of namespace slnn
