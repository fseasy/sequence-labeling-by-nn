#include <unordered_map>
#include <algorithm>
#include "token_lexicon.h"
#include "trivial/charcode/charcode_detector.h"
#include "segmenter/cws_module/cws_reader_unicode.h"
namespace slnn{
namespace segmenter{
namespace token_module{

TokenLexicon::TokenLexicon(unsigned maxlen4feature)
    :maxlen4feature(maxlen4feature)
{}

void TokenLexicon::build_inner_lexicon_from_training_data(std::ifstream &training_is)
{
    std::ifstream::iostate f_state = training_is.rdstate();
    std::ifstream::streampos f_pos = training_is.tellg();
    reader::SegmentorUnicodeReader reader_ins(training_is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(training_is));
    std::unordered_map<std::u32string, unsigned> word_cnt;
    std::vector<std::u32string> wordseq;
    while( reader_ins.read_segmented_line(wordseq) )
    {
        for( const std::u32string& word : wordseq ){ ++word_cnt[word]; }
    }
    // copy from LTP
    std::vector<unsigned> freq_list(word_cnt.size());
    // get frequency threshold
    unsigned idx = 0;
    unsigned long long total_freq = 0;
    for( std::unordered_map<std::u32string, unsigned>::const_iterator iter = word_cnt.cbegin(); iter != word_cnt.cend(); ++iter )
    {
        freq_list[idx++] = iter->second;
        total_freq += iter->second;
    }
    std::sort(freq_list.begin(), freq_list.end(), std::greater<unsigned>());
    unsigned long long truncated_total_freq = total_freq * 0.9L;
    unsigned long long current_freq = 0UL;
    freq_threshold = 0U;
    for( size_t i = 0; i < freq_list.size(); ++i )
    {
        current_freq += freq_list[i];
        if( current_freq >= truncated_total_freq ){ freq_threshold = freq_list[i]; break; }
    }
    // move word with at least 2 character and freqency greater than freq_threshold to lexicon
    word_maxlen_in_lexicon = 0;
    for( auto iter = word_cnt.cbegin(); iter != word_cnt.cend(); ++iter )
    {
        if( iter->second >= freq_threshold )
        {
            unsigned num_char = iter->first.length();
            if( num_char > 1U) 
            {
                // at least 2 characters
                inner_lexicon.insert(std::move(iter->first));
                word_maxlen_in_lexicon = std::max(word_maxlen_in_lexicon, num_char);
            }
        }
    }
    // restore the training file state
    training_is.clear();
    training_is.seekg(f_pos);
    training_is.setstate(f_state);

}

std::shared_ptr<std::vector<std::vector<Index>>> TokenLexicon::extract(const std::u32string& charseq) const
{
    unsigned seq_len = charseq.length();
    std::shared_ptr<std::vector<std::vector<Index>>> lexicon_feat(
        // 3 row, every row is the feature-abstracted sequence corresponding to the feature.
        new std::vector<std::vector<Index>>(3, std::vector<Index>(seq_len, 0))
    );
    // Max Match
    // this lexicon feature can be look as fusion Max Match Result
    for( unsigned i = 0; i < seq_len; ++i ) // index increasing one by one , instead of doing like MM which skip word 
    {
        // Attention, substr(pos, cnt) automatically processing the situation where `pos + cnt >= len(str)`
        std::u32string test_word = charseq.substr(i, word_maxlen_in_lexicon);
        while( test_word.length() > 1 )
        {
            if( inner_lexicon.count(test_word) > 0 ){ break; }
            test_word.pop_back();
        }
        unsigned wordlen = test_word.size();
        // set feature value
        // 1. word start
        (*lexicon_feat)[0][i] = wordlen - 1; // wordlen >= 1, translate to feature, we minus 1. 
        // 2. word middle
        for( unsigned mid = i + 1; mid < i + wordlen - 1; ++mid )
        {
            // wordlen >= 3 when going here. because feature value 0 assign to not going here.
            // so `wordlen - 2` to make feature value correct.
            (*lexicon_feat)[1][mid] = std::max(static_cast<Index>(wordlen - 2), (*lexicon_feat)[1][mid]); 
        }
        // 3. word end
        (*lexicon_feat)[2][i + wordlen - 1] = std::max(static_cast<Index>(wordlen - 1), (*lexicon_feat)[2][ i+wordlen-1 ]);
    }
    // at last, we should check the maxlen4feature
    Index feature_maxval = maxlen4feature; // for cast from unsigned to signed.
    for( std::vector<Index>& feat_row : *lexicon_feat )
    {
        for( Index &val : feat_row ){ val = std::min(val, feature_maxval); }
    }
    return lexicon_feat;
}


} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn
