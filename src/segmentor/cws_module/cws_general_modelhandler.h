#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_GENERAL_MODELHANDLER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_GENERAL_MODELHANDLER_H_
#include <string>
#include <vector>
#include <boost/program_options/variables_map.hpp>
namespace slnn{
namespace segmentor{
namespace modelhandler{

namespace modelhandler_inner{

template <typename SLModel>
void read_annotated_data(SLModel &slm, std::vector<typename SLModel::AnnotatedProcessedType> &out_ann_processed_data);

} // end of namespace modelhandler-inner

inline
const std::string& WordOutputDelimiter()
{
    // see Effective C++ , item 04.
    // for non-local static variable, to avoid initialization-race-condition, using local static variable.
    // that is, Singleton Pattern Design
    static std::string WordOutputDelimiterLocalStatic = "\t";
    return WordOutputDelimiterLocalStatic;
}

template <typename SLModel>
bool set_model_structure_param(SLModel &slm, const boost::program_options::variables_map &args);

template <typename SLModel>
void read_training_data(SLModel &slm, std::vector<typename SLModel::AnnotatedProcessedType> &out_training_processed_data);

template <typename SLModel>
void read_devel_data(SLModel &slm, std::vector<typename SLModel::AnnotatedProcessedType> &out_devel_processed_data);

template <typename SLModel>
void read_devel_data(SLModel &slm, std::vector<typename SLModel::UnannotatedProcessedType> &out_test_processed_data);



} // end of namespace modelhandler
} // end of namespace segmentor
} // end of namespace slnn

#endif