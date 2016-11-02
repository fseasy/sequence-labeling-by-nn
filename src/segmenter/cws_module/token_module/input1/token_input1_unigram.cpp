#include "token_input1_unigram.h"
namespace slnn{
namespace segmenter{
namespace token_module{

/**
* constructor, with an seed to init the replace randomization.
* @param seed unsigned, to init the inner LookupTableWithReplace.
*/
TokenSegmenterInput1Unigram::TokenSegmenterInput1Unigram(unsigned seed) noexcept
    :token_dict(seed, 1, 0.2F, [](const char32_t &token){ return token_module_inner::token2str(token); })
{}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn
