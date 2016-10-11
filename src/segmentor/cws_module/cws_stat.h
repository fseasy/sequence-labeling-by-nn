#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_STAT_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_STAT_H_
#include "utils/stat.hpp"
namespace slnn{
namespace segmentor{
namespace stat{

struct SegmentorStat : public BasicStat
{
    // data
    unsigned nr_token_predict;
    // interface
    SegmentorStat(bool is_predict = false);
    std::string get_stat_str(const std::string &info_header);
};

/*********************************************
 * inline interface
 *********************************************/
inline
SegmentorStat::SegmentorStat(bool is_predict)
    :BasicStat(is_predict),
    nr_token_predict(0)
{}

inline
std::string SegmentorStat::get_stat_str(const std::string &info_header)
{
    std::ostringstream str_os;
    str_os << info_header << "\n" ;
    if( !is_predict ){ str_os << "Sum E = " << get_sum_E() << "\n" ; }
    str_os << "Time cost = " << get_time_cost_in_seconds() << " s\n"
        << "Speed(tag) = " << get_speed_as_kilo_tokens_per_sencond() << " K Tags/s\n"
        << "Speed(token) = " << nr_token_predict / 1000.f / get_time_cost_in_seconds() << " K Tokens/s" ;
    return str_os.str();
}


} // end of namespace stat
} // end of namespace segmentor
} // end of namespace slnn

#endif