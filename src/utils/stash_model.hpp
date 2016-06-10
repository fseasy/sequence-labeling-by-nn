#ifndef UTILS_STASH_MODEL_HPP_
#define UTILS_STASH_MODEL_HPP_

#include <sstream>
#include <boost/log/trivial.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "cnn/cnn.h"

namespace slnn{

struct CNNModelStash
{
    float best_score;
    std::stringstream best_model_tmp_ss;
    CNNModelStash(float train_error_threshold=20.f);
    bool save_when_best(cnn::Model *best_model, float best_score);
    bool load_if_exists(cnn::Model *cnn_model);
    bool is_train_error_occurs(float cur_score);
    void set_train_error_threshold(float error_threshold);
private :
    float train_error_threshold;
};

inline
CNNModelStash::CNNModelStash(float train_error_threshold)
    :best_score(0.f),
    best_model_tmp_ss(""),
    train_error_threshold(train_error_threshold)
{}

/***
 * save model when the score is the best .
 * return : bool, 
 *               true if saved(best model), else return false;
*/
inline
bool CNNModelStash::save_when_best(cnn::Model *model, float score)
{
    if( score > best_score )
    {
        BOOST_LOG_TRIVIAL(info) << "better model has been found . stash it .";
        best_score = score;
        best_model_tmp_ss.str(""); // first , clear it's content !
        boost::archive::text_oarchive to(best_model_tmp_ss); // to construct an text_oarchive using stringstream
        to << *model;
        return true ;
    }
    else { return false; }
}

/***
* load model to cnn_model ptr if best model has been saved 
* return : bool
*              true if loaded, else return false
*/
inline
bool CNNModelStash::load_if_exists(cnn::Model *cnn_model)
{
    if( best_model_tmp_ss.rdbuf()->in_avail() != 0 )
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *cnn_model;
        return true;
    }
    else { return false ; }
}

inline
bool CNNModelStash::is_train_error_occurs(float cur_score)
{
    return (best_score - cur_score > train_error_threshold);
}

inline
void CNNModelStash::set_train_error_threshold(float error_threshold)
{
    train_error_threshold = error_threshold;
}

} // end of namespace slnn
#endif