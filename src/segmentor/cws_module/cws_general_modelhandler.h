#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_GENERAL_MODELHANDLER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_GENERAL_MODELHANDLER_H_
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/program_options/variables_map.hpp>
#include "utils/stat.hpp"
#include "cws_reader_unicode.h"
#include "cws_writer.h"
#include "token_module/cws_tag_definition.h"
#include "cws_eval.h"
#include "cws_stat.h"
#include "trivial/charcode/charcode_detector.h"
#include "utils/typedeclaration.h"
namespace slnn{
namespace segmentor{
namespace modelhandler{

namespace modelhandler_inner{

template <typename SLModel>
unsigned read_annotated_data(std::istream &is, SLModel &slm, 
    std::vector<typename SLModel::AnnotatedDataProcessedT> &out_ann_processed_data);

template <typename SLModel>
unsigned read_unannotated_data(std::istream &is, SLModel &slm, 
    std::vector<typename SLModel::UnannotatedDataProcessedT> &out_unann_processed_data,
    std::vector<typename SLModel::UnannotatedDataRawT> &out_unann_raw_data);

class TrainingUpdateRecorder
{
public:
    TrainingUpdateRecorder(float error_threshold = 20.f);
    void update_training_state(float current_score, int nr_epoch, int nr_devel_order);
    bool is_training_ok();
    void set_train_error_threshold(float error_threshold);
public: // getter
    int get_best_epoch(){ return nr_epoch_when_best; }
    int get_best_devel_order(){ return nr_devel_order_when_best; }
    float get_best_score(){ return best_score; }
private:
    float best_score;
    int nr_epoch_when_best;
    int nr_devel_order_when_best;
    float train_error_threshold;
    bool is_good;
};

} // end of namespace modelhandler-inner

inline
const std::u32string& WordOutputDelimiter()
{
    // see Effective C++ , item 04.
    // for non-local static variable, to avoid initialization-race-condition, using local static variable.
    // that is, Singleton Pattern Design
    static std::u32string WordOutputDelimiterLocalStatic = U"\t";
    return WordOutputDelimiterLocalStatic;
}

template <typename SLModel>
void set_model_structure_param(SLModel &slm, const boost::program_options::variables_map &args);

template <typename SLModel>
void read_training_data(std::istream &is, SLModel &slm, std::vector<typename SLModel::AnnotatedDataProcessedT> &out_training_processed_data);

template <typename SLModel>
void read_devel_data(std::istream &is, SLModel &slm, std::vector<typename SLModel::AnnotatedDataProcessedT> &out_devel_processed_data);

template <typename SLModel>
void read_test_data(std::istream &is, SLModel &slm, 
    std::vector<typename SLModel::UnannotatedDataProcessedT> &out_test_processed_data,
    std::vector<typename SLModel::UnannotatedDataRawT> &out_test_raw_data);

template <typename SLModel>
void build_model(SLModel &slm);

template <typename SLModel, typename TrainingOpts>
void train(SLModel &slm,
    const std::vector<typename SLModel::AnnotatedDataProcessedT> &training_data,
    const std::vector<typename SLModel::AnnotatedDataProcessedT> &devel_data,
    const TrainingOpts &opts);

template <typename SLModel>
float devel(SLModel &slm,
    const std::vector<typename SLModel::AnnotatedDataProcessedT> &devel_data);

template <typename SLModel>
void predict(SLModel &slm,
    std::istream &is,
    std::ostream &os);


/*********************************************************
 * Inline Implementation
 *********************************************************/

namespace modelhandler_inner{

template <typename SLModel>
unsigned read_annotated_data(std::istream &is, SLModel &slm, std::vector<typename SLModel::AnnotatedDataProcessedT> &out_ann_processed_data)
{
    using std::swap;
    reader::SegmentorUnicodeReader reader_ins(is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(is));
    std::vector<typename SLModel::AnnotatedDataProcessedT> dataset;
    unsigned detected_line_cnt = reader_ins.count_line();
    dataset.reserve(detected_line_cnt);
    typename SLModel::AnnotatedDataRawT wordseq;
    unsigned readline_cnt = 0,
        report_cnt = detected_line_cnt / 5;
    while( reader_ins.read_segmented_line(wordseq) )
    {
        if( wordseq.empty() ){ continue; }
        typename SLModel::AnnotatedDataProcessedT processed_data;
        slm.get_token_module()->process_annotated_data(wordseq, processed_data);
        dataset.push_back(std::move(processed_data));
        ++readline_cnt;
        if( report_cnt && readline_cnt % report_cnt == 0 )
        {
            std::cerr << "- read instance : " << readline_cnt << " [" << readline_cnt / report_cnt << "/5]\n";
        }
    }
    swap(out_ann_processed_data, dataset);
    return readline_cnt;
}

template <typename SLModel>
unsigned read_unannotated_data(std::istream &is, SLModel &slm,
    std::vector<typename SLModel::UnannotatedDataProcessedT> &out_unann_processed_data,
    std::vector<typename SLModel::UnannotatedDataRawT> &out_unann_raw_data)
{
    using std::swap;
    reader::SegmentorUnicodeReader reader_ins(is,
        charcode::EncodingDetector::get_detector()->detect_and_set_encoding(is)); // BUG! TODO
    std::vector<typename SLModel::UnannotatedDataProcessedT> dataset;
    std::vector<typename SLModel::UnannotatedDataRawT> raw_dataset;
    unsigned detected_line_cnt = reader_ins.count_line();
    dataset.reserve(detected_line_cnt);
    raw_dataset.reserve(detected_line_cnt);
    typename SLModel::UnannotatedDataRawT charseq;
    unsigned readline_cnt = 0,
        report_cnt = detected_line_cnt / 5;
    while( reader_ins.readline(charseq) )
    {
        typename SLModel::UnannotatedDataProcessedT processed_data;
        slm.get_token_module()->process_unannotated_data(charseq, processed_data);
        dataset.push_back(std::move(processed_data));
        raw_dataset.push_back(std::move(charseq));
        ++readline_cnt;
        if( report_cnt && readline_cnt % report_cnt == 0 )
        {
            std::cerr << "- read instance : " << readline_cnt << " [" << readline_cnt / report_cnt << "/5]\n";
        }
    }
    swap(out_unann_processed_data, dataset);
    swap(out_unann_raw_data, raw_dataset);
    return readline_cnt;
}


inline
void TrainingUpdateRecorder::set_train_error_threshold(float error_threshold)
{
    train_error_threshold = error_threshold;
}

inline 
void TrainingUpdateRecorder::update_training_state(float cur_score, int nr_epoch, int nr_devel_order)
{
    is_good = ((best_score - cur_score) < train_error_threshold);
    if( cur_score > best_score )
    {
        nr_epoch_when_best = nr_epoch;
        nr_devel_order_when_best = nr_devel_order;
        best_score = cur_score;
    }
}

inline 
bool TrainingUpdateRecorder::is_training_ok()
{
    return is_good;
}

} // end of namespce modelhandler-inner


template <typename SLModel>
void set_model_structure_param(SLModel &slm, const boost::program_options::variables_map &args)
{
    slm.set_model_structure_from_outer(args);
}

template <typename SLModel>
void read_training_data(std::istream &is, SLModel &slm, std::vector<typename SLModel::AnnotatedDataProcessedT> &out_training_processed_data)
{
    std::cerr << "+ Process training data.\n";
    unsigned line_cnt = modelhandler_inner::read_annotated_data(is, slm, out_training_processed_data);
    std::cerr << "= Training data processed done. (line count: " << line_cnt << ", instance number: " <<
        out_training_processed_data.size() << ")\n";
    slm.finish_read_training_data();
}

template <typename SLModel>
void read_devel_data(std::istream &is, SLModel &slm, std::vector<typename SLModel::AnnotatedDataProcessedT> &out_devel_processed_data)
{
    std::cerr << "+ Process devel data.\n";
    unsigned line_cnt = modelhandler_inner::read_annotated_data(is, slm, out_devel_processed_data);
    std::cerr << "= Devel data processed done. (line count: " << line_cnt << ", instance number: " 
        << out_devel_processed_data.size() << ")\n";
}

template <typename SLModel>
void read_test_data(std::istream &is, SLModel &slm, 
    std::vector<typename SLModel::UnannotatedDataProcessedT> &out_test_processed_data,
    std::vector<typename SLModel::UnannotatedDataRawT> &out_test_raw_data)
{
    std::cerr << "+ Process test data.\n";
    unsigned line_cnt = modelhandler_inner::read_unannotated_data(is, slm, out_test_processed_data, out_test_raw_data);
    std::cerr << "= Test data processed done. (line count: " << line_cnt << ", instance number: "
        << out_test_processed_data.size() << ")\n";
}

template <typename SLModel>
void build_model(SLModel &slm)
{
    slm.build_model_structure();
}

template <typename SLModel, typename TrainingOpts>
void train(SLModel &slm,
    const std::vector<typename SLModel::AnnotatedDataProcessedT> &training_data,
    const std::vector<typename SLModel::AnnotatedDataProcessedT> &devel_data,
    const TrainingOpts &opts)
{
    unsigned nr_samples = training_data.size();
    std::cerr << "+ Train at " << nr_samples << " instances .\n";
    
    slm.get_nn()->set_update_method(opts.training_update_method);

    modelhandler_inner::TrainingUpdateRecorder update_recorder;
    auto do_devel_in_training = [&devel_data, &slm, &update_recorder](int nr_epoch, int nr_devel_order) 
    {
        // function : 1. devel; 2. stash model when best; 3. update training  state.
        float f1 = devel(slm, devel_data);
        slm.get_nn()->stash_model_when_best(f1);
        update_recorder.update_training_state(f1, nr_epoch, nr_devel_order);
    };

    // for randomly select instance
    std::vector<unsigned> access_order(nr_samples);
    for( unsigned i = 0; i < nr_samples; ++i ) access_order[i] = i;

    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    for( unsigned nr_epoch = 1; nr_epoch <= opts.max_epoch ; ++nr_epoch )
    {
        std::cerr << "++ Epoch " << nr_epoch << "/" << opts.max_epoch << " start. \n";
        // shuffle samples by random access order
        std::shuffle(access_order.begin(), access_order.end(), *slm.get_mt19937_rng());

        // For loss , accuracy , time cost report
        BasicStat training_stat_per_epoch;
        training_stat_per_epoch.start_time_stat();

        int nr_devel_order = 0; // for record
        // train for every Epoch 
        for( unsigned i = 0; i < nr_samples; ++i )
        {
            unsigned access_idx = access_order[i];
            const  typename SLModel::AnnotatedDataProcessedT &instance = training_data[access_idx];
            // GO
            slm.build_training_graph(instance);
            slnn::type::real loss = slm.get_nn()->forward_as_scalar();
            slm.get_nn()->backward();
            slm.get_nn()->update(opts.training_update_scale);
            // record loss
            training_stat_per_epoch.loss += loss;
            training_stat_per_epoch.total_tags += instance.size() ;
            if( 0 == (i + 1) % opts.trivial_report_freq && false) // Report 
            {
                std::string trivial_header = std::to_string(i + 1) + " instances have been trained.";
                std::cerr << training_stat_per_epoch.get_stat_str(trivial_header) << "\n";
            }
            ++line_cnt_for_devel;
            // do devel at every `do_devel_freq` and if update error, just exit the current process
            if( 0 == line_cnt_for_devel % opts.do_devel_freq )
            {
                line_cnt_for_devel = 0; // clear
                do_devel_in_training(nr_epoch, ++nr_devel_order);
                if( !update_recorder.is_training_ok() ){ break;  }
            }
        }

        // End of an epoch 
        // 1. update epoch
        slm.get_nn()->update_epoch();
        // 2. end timing 
        training_stat_per_epoch.end_time_stat();
        // 3. output info at end of every eopch
        std::ostringstream tmp_sos;
        tmp_sos << "-- Epoch " << nr_epoch << "/" << opts.max_epoch << " finished .\n";
        std::cerr << training_stat_per_epoch.get_stat_str(tmp_sos.str()) << "\n";
        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
        // do validation at every ends of epoch
        if( update_recorder.is_training_ok() )
        {
            std::cerr << "-- Do validation at every ends of epoch .\n";
            do_devel_in_training(nr_epoch, -1); // -1 stands for the end of the epoch.
        }
        // check angin! (previous devel process may change the state.)
        if( !update_recorder.is_training_ok() ){ break; }
    }
    if( !update_recorder.is_training_ok() )
    { 
        std::cerr << "! Gradient may have been updated error ! Exit ahead of time.\n" ; 
    }
    std::cerr << "= Training finished. Time cost:" << total_time_cost_in_seconds << "s, \n"
        << "| best score(F1): " << update_recorder.get_best_score() << "%, \n"
        << "| at epoch: " << update_recorder.get_best_epoch() << ", \n"
        << "| devel order: " << update_recorder.get_best_devel_order() << "\n"
        << "= - - - - -\n";
}

template <typename SLModel>
float devel(SLModel &slm, const std::vector<typename SLModel::AnnotatedDataProcessedT> &devel_data)
{
    unsigned nr_samples = devel_data.size();
    std::cerr << "+ Validation at " << nr_samples << " instances.\n";

    stat::SegmentorStat stat(true);
    stat.start_time_stat();
    eval::SegmentorEval eval_ins;
    eval_ins.start_eval();
    for( unsigned access_idx = 0; access_idx < nr_samples; ++access_idx )
    {
        const typename SLModel::AnnotatedDataProcessedT &instance = devel_data[access_idx];
        std::vector<Index> pred_tagseq = slm.predict(
            *slm.get_token_module()->extract_unannotated_data_from_annotated_data(instance));
        eval_ins.eval_iteratively(*instance.ptagseq, pred_tagseq);
    }
    stat.end_time_stat();
    eval::EvalResultT eval_result = eval_ins.end_eval();
    stat.nr_token_predict = eval_result.nr_token_predict;
    stat.total_tags = eval_result.nr_tag;
    std::ostringstream tmp_sos;
    tmp_sos << "= Validation finished. \n"
        << "| Acc = " << eval_result.acc << "% , P = " << eval_result.p 
        << "% , R = " << eval_result.r << "% , F1 = " << eval_result.f1 << "%";
    std::cerr << stat.get_stat_str(tmp_sos.str()) << "\n";
    return eval_result.f1;
}

template <typename SLModel>
void predict(SLModel &slm,
    std::istream &is,
    std::ostream &os)
{
    std::vector<typename SLModel::UnannotatedDataProcessedT> test_data;
    std::vector<typename SLModel::UnannotatedDataRawT> test_raw_data;
    read_test_data(is, slm, test_data, test_raw_data);
    std::cerr << "+ Do prediction on " << test_data.size() << " instances .";
    BasicStat stat(true);
    stat.start_time_stat();
    writer::SegmentorWriter writer_ins(os, charcode::EncodingDetector::get_detector()->get_encoding(), WordOutputDelimiter());
    for (unsigned int i = 0; i < test_data.size(); ++i)
    {
        typename SLModel::UnannotatedDataRawT &raw_instance = test_raw_data[i];
        typename SLModel::UnannotatedDataProcessedT &instance = test_data[i];
        if (0 == raw_instance.size())
        {
            writer_ins.write({}, {});
            continue;
        }
        std::vector<Index> pred_tagseq = slm.predict(instance);
        writer_ins.write(raw_instance, pred_tagseq);
        stat.total_tags += pred_tagseq.size() ;
    }
    stat.end_time_stat() ;
    std::cerr << stat.get_stat_str("+ Predict done.") << "\n" ;
}


} // end of namespace modelhandler
} // end of namespace segmentor
} // end of namespace slnn

#endif
