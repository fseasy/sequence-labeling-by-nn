#include "context_feature.h"

namespace slnn{

ContextFeature::ContextFeature(unsigned context_size, DictWrapper &word_dict_wrapper)
    :ContextSize(context_size),
    ContextLeftSize(context_size / 2),
    ContextRightSize(ContextSize - ContextLeftSize),
    word_dict_wrapper(word_dict_wrapper)
{}

}