#include "token_input1_bigram.h"

namespace slnn{
namespace segmenter{
namespace token_module{

std::u32string TokenSegmenterInput1Bigram::EOS_REPR = U"<EOS>";

TokenSegmenterInput1Bigram::TokenSegmenterInput1Bigram(unsigned seed) noexcept
    :token_dict(seed, 1, 0.2F, [](const std::u32string &token) -> std::string{
    return input1_bigram_token_module_inner::token2str(token); })
{}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn