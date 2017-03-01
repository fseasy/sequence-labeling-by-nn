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
        iss << word_seq[0] << TEST_WORD_POS_DELIM << pos_tag_seq[0];
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        iss << U" " << word_seq[i] << TEST_WORD_POS_DELIM << pos_tag_seq[i];
    }
    return iss.str();
}

std::u32string AnnotatedInstance::to_string()
{
    std::basic_stringstream<char32_t> iss;
    if( size() > 0 )
    {
        iss << word_seq[0] << TRAIN_WORD_POS_DELIM << pos_tag_seq[0]
            << POS_NER_DELIM << ner_tag_seq[0];
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        iss << U" " << word_seq[i] << pos_tag_seq[i]
            << POS_NER_DELIM << ner_tag_seq[i];
    }
    return iss.str();
}

AnnotatedInstance line2annotated_instance(const std::u32string& uline)
{
    AnnotatedInstance instance;
    // WORD_POS#NER\tWORD_POS#NER
    auto get_token = [](const std::u32string &piece, std::u32string& word,
        std::u32string& pos_tag, std::u32string& ner_tag,
        char32_t word_pos_delim=TRAIN_WORD_POS_DELIM,
        char32_t pos_ner_delim=POS_NER_DELIM)
    {
        auto underline_pos = piece.rfind(word_pos_delim),
            sharp_pos = piece.rfind(pos_ner_delim);
        if( underline_pos == std::u32string::npos ||
            sharp_pos == std::u32string::npos ||
            sharp_pos < underline_pos )
        {
            throw std::runtime_error("ill-formated annotated instance");
        }
        word = piece.substr(0, underline_pos);
        pos_tag = piece.substr(underline_pos + 1, sharp_pos - underline_pos - 1);
        ner_tag = piece.substr(sharp_pos + 1);
    };
    std::size_t piece_spos = 0U,
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    while( piece_epos != std::u32string::npos )
    {
        std::u32string word, pos_tag, ner_tag;
        get_token(uline.substr(piece_spos, piece_epos - piece_spos), word, pos_tag, ner_tag);
        instance.push_back(std::move(word), std::move(pos_tag), std::move(ner_tag));
        piece_spos = piece_epos + 1;
        // if piece_spos >= size(), always return std::u32string::npos
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    }
    // the last part
    std::u32string word, pos_tag, ner_tag;
    get_token(uline.substr(piece_spos), word, pos_tag, ner_tag);
    instance.push_back(std::move(word), std::move(pos_tag), std::move(ner_tag));
    return instance;
}

UnannotatedInstance line2unannotated_instance(const std::u32string &uline)
{
    UnannotatedInstance instance;
    // WORD_POS#NER\tWORD_POS#NER
    auto get_token = [](const std::u32string &piece, std::u32string& word,
        std::u32string& pos_tag,
        char32_t word_pos_delim=TEST_WORD_POS_DELIM)
    {
        auto underline_pos = piece.rfind(word_pos_delim);
        if( underline_pos == std::u32string::npos)
        {
            throw std::runtime_error("ill-formated annotated instance");
        }
        word = piece.substr(0, underline_pos);
        pos_tag = piece.substr(underline_pos + 1);
    };
    std::size_t piece_spos = 0U,
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    while( piece_epos != std::u32string::npos )
    {
        std::u32string word, pos_tag;
        get_token(uline.substr(piece_spos, piece_epos - piece_spos), word, pos_tag);
        instance.push_back(std::move(word), std::move(pos_tag));
        piece_spos = piece_epos + 1;
        // if piece_spos >= size(), always return std::u32string::npos
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    }
    // the last part
    std::u32string word, pos_tag;
    get_token(uline.substr(piece_spos), word, pos_tag);
    instance.push_back(std::move(word), std::move(pos_tag));
    return instance;
}

TokenDict build_token_dict(const std::vector<AnnotatedInstance>&)
{
    // !TODO
    TokenDict token_dict;
    return token_dict;
}

} // end of namespace token-module
} // end of namespace ner
} // end of namespace slnn