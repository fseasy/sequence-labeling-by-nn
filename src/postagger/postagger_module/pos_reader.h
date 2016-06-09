#ifndef POS_POS_MODULE_POS_READER_HPP_
#define POS_POS_MODULE_POS_READER_HPP_

#include <fstream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "utils/typedeclaration.h"
#include "utils/reader.hpp"

namespace slnn{

class POSReader : public Reader
{
public:
    static const char* PosDataDelimiter ;
    static const char* WordPosDelimiter;
public:
    
    POSReader(std::istream &is);
    bool readline(Seq &sent, Seq &postag); // training data
    bool readline(Seq &sent); // devel data
};


inline
POSReader::POSReader(std::istream &is)
    :Reader(is)
{}

inline
bool POSReader::readline(Seq &sent, Seq &postag_seq)
{
    using std::swap;
    Seq tmp_sent,
        tmp_tag_seq;
    std::string line;
    bool is_good = getline(is, line).good() ;
    if( ! is_good ) return false;
    std::vector<std::string> strpair_cont;
    boost::algorithm::split(strpair_cont, line, boost::is_any_of(PosDataDelimiter));
    size_t pair_len = strpair_cont.size();
    tmp_sent.resize(pair_len);
    tmp_tag_seq.resize(pair_len);
    for( size_t i = 0; i < pair_len; ++i )
    {
        const std::string &str_pair = strpair_cont[i];
        std::string::size_type  delim_pos = str_pair.rfind(WordPosDelimiter);
        assert(delim_pos != std::string::npos);
        tmp_sent[i] = str_pair.substr(0, delim_pos);
        tmp_tag_seq[i] = str_pair.substr(delim_pos + 1);
    }
    swap(sent, tmp_sent);
    swap(postag_seq, tmp_tag_seq);
    return true ;
}

inline
bool POSReader::readline(Seq &sent)
{
    using std::swap;
    Seq word_cont;
    std::string line;
    bool is_good = getline(is, line).good() ;
    if( !is_good ) return false;
    boost::algorithm::split(word_cont, line, boost::is_any_of(PosDataDelimiter));
    swap(sent, word_cont);
    return true;
}


} // end of namespace slnn
#endif
