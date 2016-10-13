/**
 * tag_utility.h, including the utilities for tag system.
 */

#ifndef SLNN_SEGMENTOR_CWS_MODULE_TOKEN_MODULE_TAG_UTILITY_H_
#define SLNN_SEGMENTOR_CWS_MODULE_TOKEN_MODULE_TAG_UTILITY_H_
#include "cws_tag_definition.h"

namespace slnn{
namespace segmentor{
namespace token_module{



/****************************************************
 * Inline Implementation
 ****************************************************/
Index select_best_tag_constrained(std::vector<cnn::real> &dist, size_t time, Index pre_time_tag_id)
{
    cnn::real max_prob = std::numeric_limits<cnn::real>::lowest();
    Index tag_with_max_prob = Tag::TAG_NONE_ID;
    constexpr Index max_tag_id = TAG_SIZE - 1;
    for( Index tag_id = 0; tag_id <= max_tag_id; ++tag_id )
    {
        if( !can_emit(time, tag_id) ){ continue; }
        if( time > 0 && !can_trans(pre_time_tag_id, tag_id) ){ continue; }
        if( dist[tag_id] >= max_prob )
        {
            tag_with_max_prob = tag_id;
            max_prob = dist[tag_id];
        }
    }
    // assert(tag_with_max_prob != STATIC_NONE_ID);
    return tag_with_max_prob;
}

}
}
} // end of namespace slnn

#endif