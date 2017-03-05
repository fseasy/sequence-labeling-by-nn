#include <sstream>
#include "token_module.h"
namespace slnn{
namespace ner{
namespace token_module{

Str UnannotatedInstance::to_string() const
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

Str AnnotatedInstance::to_string() const
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

AnnotatedInstance line2annotated_instance(const Str& uline)
{
    AnnotatedInstance instance;
    // WORD_POS#NER\tWORD_POS#NER
    auto get_token = [](const Str &piece, Str& word,
        Str& pos_tag, Str& ner_tag,
        char32_t word_pos_delim=TRAIN_WORD_POS_DELIM,
        char32_t pos_ner_delim=POS_NER_DELIM)
    {
        auto underline_pos = piece.rfind(word_pos_delim),
            sharp_pos = piece.rfind(pos_ner_delim);
        if( underline_pos == Str::npos ||
            sharp_pos == Str::npos ||
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
    while( piece_epos != Str::npos )
    {
        Str word, pos_tag, ner_tag;
        get_token(uline.substr(piece_spos, piece_epos - piece_spos), word, pos_tag, ner_tag);
        instance.push_back(std::move(word), std::move(pos_tag), std::move(ner_tag));
        piece_spos = piece_epos + 1;
        // if piece_spos >= size(), always return Str::npos
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    }
    // the last part
    Str word, pos_tag, ner_tag;
    get_token(uline.substr(piece_spos), word, pos_tag, ner_tag);
    instance.push_back(std::move(word), std::move(pos_tag), std::move(ner_tag));
    return instance;
}

UnannotatedInstance line2unannotated_instance(const Str &uline)
{
    UnannotatedInstance instance;
    // WORD_POS#NER\tWORD_POS#NER
    auto get_token = [](const Str &piece, Str& word,
        Str& pos_tag,
        char32_t word_pos_delim=TEST_WORD_POS_DELIM)
    {
        auto underline_pos = piece.rfind(word_pos_delim);
        if( underline_pos == Str::npos)
        {
            throw std::runtime_error("ill-formated annotated instance");
        }
        word = piece.substr(0, underline_pos);
        pos_tag = piece.substr(underline_pos + 1);
    };
    std::size_t piece_spos = 0U,
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    while( piece_epos != Str::npos )
    {
        Str word, pos_tag;
        get_token(uline.substr(piece_spos, piece_epos - piece_spos), word, pos_tag);
        instance.push_back(std::move(word), std::move(pos_tag));
        piece_spos = piece_epos + 1;
        // if piece_spos >= size(), always return Str::npos
        piece_epos = uline.find(PIECE_DELIM, piece_spos);
    }
    // the last part
    Str word, pos_tag;
    get_token(uline.substr(piece_spos), word, pos_tag);
    instance.push_back(std::move(word), std::move(pos_tag));
    return instance;
}

std::shared_ptr<TokenDict >
build_token_dict(const std::vector<AnnotatedInstance>& instance_list)
{
    TokenDict token_dict;
    for( const auto& instance : instance_list )
    {
        for( const auto& word : instance.word_seq )
        {
            token_dict.word_dict.convert(word);
        }
        for( const auto& pos : instance.pos_tag_seq )
        {
            token_dict.pos_tag_dict.convert(pos);
        }
        for( const auto& ner : instance.ner_tag_seq )
        {
            token_dict.ner_tag_dict.convert(ner);
        }
    }
    token_dict.freeze_and_set_unk();
    return std::make_shared<TokenDict>(token_dict);
}

std::shared_ptr<WordFeatInfo>
build_word_feat_info(const TokenDict& dict, const std::vector<InstanceFeature>& feat_list)
{
    WordFeatInfo info;
    std::size_t word_num = dict.word_num_with_unk();
    info.word_cnt_lookup.resize(word_num, 0U);
    for( size_t i = 0; i < feat_list.size(); ++i )
    {
        for( InstanceFeature::FeatIndex word_id : *(feat_list[i].word_feat) )
        {
            ++info.word_cnt_lookup[word_id];
        }
    }
    info.word_unk_index = dict.word_dict.get_unk_idx();
    return std::make_shared<WordFeatInfo>(info);
}


Str InstanceFeature::to_string() const
{
    std::basic_stringstream<char32_t> oss;
    auto out1pair = [&oss, this](std::size_t pos)
    {
        oss << std::to_string(this->word_feat->at(pos)).c_str() << TEST_WORD_POS_DELIM
            << std::to_string(this->pos_tag_feat->at(pos)).c_str();
    };
    if( size() > 0U )
    {
        out1pair(0);
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        oss << U' ';
        out1pair(i);
    }
    return oss.str();
}

Str InstanceFeature::to_char_string(const TokenDict& dict) const
{
    std::basic_stringstream<char32_t> oss;
    auto out1pair = [&dict, &oss, this](std::size_t pos)
    {
        Str word;
        InstanceFeature::FeatIndex word_feat = this->word_feat->at(pos);
        if( word_feat == dict.word_dict.get_unk_idx() )
        {
            word = U"_UNK_";
        }
        else{ word = dict.word_dict.convert_ban_unk(word_feat); }
        oss << word
            << TEST_WORD_POS_DELIM
            << dict.pos_tag_dict.convert_ban_unk(this->pos_tag_feat->at(pos));
    };
    if( size() > 0U )
    {
        out1pair(0U);
    }
    for( std::size_t i = 1U; i < size(); ++i )
    {
        oss << U' ';
        out1pair(i);
    }
    return oss.str();
}

NerTagIndexSeq ner_seq2ner_index_seq(const StrSeq& ner_seq, const TokenDict& dict)
{
    std::size_t len = ner_seq.size();
    NerTagIndexSeq ner_index_seq(len);
    for( std::size_t i = 0; i < len; ++i )
    {
        ner_index_seq[i] = dict.ner_tag_dict.convert(ner_seq[i]);
    }
    return ner_index_seq;
}

StrSeq ner_index_seq2ner_seq(const NerTagIndexSeq& index_seq, const TokenDict &dict)
{
    std::size_t len = index_seq.size();
    StrSeq ner_tag_seq(len);
    for( std::size_t i = 0; i < len; ++i )
    {
        ner_tag_seq[i] = dict.ner_tag_dict.convert_ban_unk(index_seq[i]);
    }
    return ner_tag_seq;
}

} // end of namespace token-module
} // end of namespace ner
} // end of namespace slnn