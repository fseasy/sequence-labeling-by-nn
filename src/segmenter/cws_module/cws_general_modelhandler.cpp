#include "cws_general_modelhandler.h"

namespace slnn{
namespace segmenter{
namespace modelhandler{

namespace modelhandler_inner{

TrainingUpdateRecorder::TrainingUpdateRecorder(float error_threshold)
    :best_score(0.f),
    nr_epoch_when_best(0),
    nr_devel_order_when_best(0),
    train_error_threshold(error_threshold),
    is_good(true)
{}



} // end of namespce modelhandler-inner

} // end of namespace modelhandler
} // end of namespace segmenter
} // end of namespace slnn
