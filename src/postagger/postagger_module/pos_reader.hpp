#ifndef POS_POS_MODULE_POS_READER_HPP_
#define POS_POS_MODULE_POS_READER_HPP_

#include <fstream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "utils/typedeclaration.h"

namespace slnn{

class POSReader
{
public:
    static const char* PosDataDelimiter ;
public:
    
    POSReader(std::ifstream &is);
    bool good();
    bool readline(Seq &sent, Seq &postag); // training data
    bool readline(Seq &sent); // devel data

private :
    std::ifstream &is;
};


const char* POSReader::PosDataDelimiter = "\t";

POSReader::POSReader(std::ifstream &is)
    :is(is)
{}

bool POSReader::good()
{
    return is.good();
}

bool POSReader::readline(Seq &sent, Seq &postag_seq)
{
    using std::swap;
    bool file_good = is.good();
    if( !file_good ) return file_good;
    Seq tmp_sent,
        tmp_tag_seq;
    std::string line;
    getline(is, line);
    std::vector<std::string> strpair_cont;
    boost::algorithm::split(strpair_cont, line, boost::is_any_of(PosDataDelimiter));
    size_t pair_len = strpair_cont.size();
    tmp_sent.resize(pair_len);
    tmp_tag_seq.resize(pair_len);
    for( size_t i = 0; i < pair_len; ++i )
    {
        const std::string &str_pair = strpair_cont[i];
        std::string::size_type  delim_pos = str_pair.rfind("_");
        assert(delim_pos != std::string::npos);
        tmp_sent[i] = str_pair.substr(0, delim_pos);
        tmp_tag_seq[i] = str_pair.substr(delim_pos + 1);
    }
    swap(sent, tmp_sent);
    swap(postag_seq, tmp_tag_seq);
    return file_good;
}

bool POSReader::readline(Seq &sent)
{
    using std::swap;
    bool file_good = is.good();
    if( !file_good ) return file_good;
    Seq word_cont;
    std::string line;
    getline(is, line);
    boost::algorithm::split(word_cont, line, boost::is_any_of(PosDataDelimiter));
    swap(sent, word_cont);
    return file_good;
}

} // end of namespace slnn
#endif