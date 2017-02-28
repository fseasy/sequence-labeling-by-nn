#include <sstream>
#include "token_module.h"
namespace slnn{
namespace ner{
namespace token_module{

std::u32string UnannotatedInstance::to_string()
{
    std::basic_stringstream<char32_t> iss;
    if( size() > 0 )
    {
        iss << word_seq[0] << DEVEL_WORD_POS_DELIM << pos_tag_seq[0];
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        iss << U" " << word_seq[i] << DEVEL_WORD_POS_DELIM << pos_tag_seq[i];
    }
    return iss.str();
}

std::u32string AnnotatedInstance::to_string()
{
    std::basic_stringstream<char32_t> iss;
    std::cout << size() << " ";
    if( size() > 0 )
    {
        iss << word_seq[0] << TRAIN_WORD_POS_DELIM << pos_tag_seq[0]
            << POS_NER_DELIM << ner_tag_seq[0];
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        iss << U"LLLLL" << word_seq[i] << pos_tag_seq[i]
            << POS_NER_DELIM << ner_tag_seq[i];
    }
    return iss.str();
}



} // end of namespace token-module
} // end of namespace ner
} // end of namespace slnn