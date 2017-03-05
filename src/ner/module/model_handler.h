#ifndef SLNN_NER_MODEL_HANDLER_INCLUDE_
#define SLNN_NER_MODEL_HANDLER_INCLUDE_

#include <iostream>
#include <vector>
#include <string>
#include <array>

#include "utils/typedeclaration.h"
#include "utils/stat.hpp"

namespace slnn{
namespace ner{
namespace handler{

namespace inner{

class TrainingUpdateRecorder
{
public:
    TrainingUpdateRecorder(float error_threshold = 20.f);
    void update_training_state(float current_score, int nr_epoch, int nr_devel_order);
    bool is_training_ok();
    void set_train_error_threshold(float error_threshold);
    void write_record_list(std::ostream &os);
public: // getter
    int get_best_epoch(){ return nr_epoch_when_best; }
    int get_best_devel_order(){ return nr_devel_order_when_best; }
    float get_best_score(){ return best_score; }
    std::vector<std::tuple<float, int, int>> get_record_list(){ return record_list; }
private:
    float best_score;
    int nr_epoch_when_best;
    int nr_devel_order_when_best;
    float train_error_threshold;
    bool is_good;
    std::vector<std::tuple<float, int, int>> record_list;
};

void write_record_list(std::ostream &os, const std::vector<std::tuple<float, int, int>> &record_list);

} // end of namespace inner


inline
const std::u32string& WordOutputDelimiter()
{
    // see Effective C++ , item 04.
    // for non-local static variable, to avoid initialization-race-condition, using local static variable.
    // that is, Singleton Pattern Design
    static std::u32string WordOutputDelimiterLocalStatic = U"\t";
    return WordOutputDelimiterLocalStatic;
}

template <typename FeatType, typename NnModel, typename TokenDict, typename TrainingOpts>
std::vector<std::tuple<float, int, int>>
train(NnModel &m, TokenDict& dict,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &training_data,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &devel_data,
    const TrainingOpts &opts);

template <typename FeatType, typename NnModel, typename TokenDict>
float devel(NnModel &m, TokenDict& dict,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &devel_data,
    const std::string& conlleval_script_path="./ner_eval.sh");

template <typename NnModel, typename TokenDict>
void predict(NnModel &m, TokenDict& dict,
    std::istream &is,
    std::ostream &os);


/***********
 * inline/template implementation
 *********************/

namespace inner{



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
    record_list.push_back(std::make_tuple(cur_score, nr_epoch, nr_devel_order));
}

inline 
bool TrainingUpdateRecorder::is_training_ok()
{
    return is_good;
}

inline
void TrainingUpdateRecorder::write_record_list(std::ostream &os)
{
    // no override when different namespace
    inner::write_record_list(os, record_list);
}

inline
void write_record_list(std::ostream &os, const std::vector<std::tuple<float, int, int>> &record_list)
{
    os << "F1" << "\t" << "devel-epoch" << "\t" 
        << "devel-order" << "\n";
    for(const std::tuple<float, int, int> &record : record_list )
    {
        os << std::get<0>(record) << "\t" << std::get<1>(record) << "\t"
            << std::get<2>(record) << "\n";
    }
}


}


template <typename FeatType, typename NnModel, typename TokenDict, typename TrainingOpts>
std::vector<std::tuple<float, int, int>>
train(NnModel &m, TokenDict& dict,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &training_data,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &devel_data,
    const TrainingOpts& opts)
{
    unsigned nr_samples = training_data.size();
    std::cerr << "+ Train at " << nr_samples << " instances .\n";
    std::cerr << "++ Training info: \n"
        << "|  training update method(" << opts.training_update_method <<"),\n"
        << "|  learning rate(" << opts.learning_rate << ") eta decay(" << opts.eta_decay << "_\n"
        << "|  training update scale(" << opts.training_update_scale << "), "
        << "half decay period (" <<  opts.scale_half_decay_period  << " epochs)\n"
        << "|  max epoch(" << opts.max_epoch << "), devel frequence(" << opts.do_devel_freq << ")\n"
        << "== - - - - -\n";
    m.set_update_method(opts.training_update_method);
    m.set_optimizer_params(opts.learning_rate, opts.eta_decay);
    modelhandler_inner::TrainingUpdateRecorder update_recorder;
    auto do_devel_in_training = [&devel_data, &m, &dict, &update_recorder, &opts](int nr_epoch, int nr_devel_order) 
    {
        // function : 1. devel; 2. stash model when best; 3. update training  state.
        float f1 = devel(m, dict, devel_data, opts.conll_eval_script_path);
        m.stash_model_when_best(f1);
        update_recorder.update_training_state(f1, nr_epoch, nr_devel_order);
    };
    float actual_scale = opts.training_update_scale;
    auto update_epoch = [&opts, &m, &actual_scale](int nr_epoch)
    {
        if( nr_epoch != 0 && nr_epoch % opts.scale_half_decay_period == 0 ){ actual_scale /= 2.f; }
        m.update_epoch();
        std::cerr << "-- Update epoch. learning rate currently is " << m.get_current_learning_rate()
            << "\n";
    };
    // for randomly select instance
    std::vector<unsigned> access_order(nr_samples);
    for( unsigned i = 0; i < nr_samples; ++i ) access_order[i] = i;

    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    const std::vector<FeatType>& feat_list = training_data.first;
    const std::vector<IndexSeq>& gold_tag_seq_list = training_data.second;
    for( unsigned nr_epoch = 1; nr_epoch <= opts.max_epoch ; ++nr_epoch )
    {
        std::cerr << "++ Epoch " << nr_epoch << "/" << opts.max_epoch << " start. \n";
        // shuffle samples by random access order
        std::shuffle(access_order.begin(), access_order.end(), *m.get_mt19937_rng());

        // For loss , accuracy , time cost report
        BasicStat training_stat_per_epoch;
        training_stat_per_epoch.start_time_stat();

        int nr_devel_order = 0; // for record
                                // train for every Epoch 
        for( unsigned i = 0; i < nr_samples; ++i )
        {
            unsigned access_idx = access_order[i];
            const FeatType& feat = feat_list[access_idx];
            const IndexSeq& tag_seq = gold_tag_seq_list[access_idx];
            // GO
            typename NnModel::NnExprT loss_expr = m.build_training_graph(feat, tag_seq);
            slnn::type::real loss = m.as_scalar(m.forward(loss_expr));
            m.backward(loss_expr);
            m.update(actual_scale);
            // record loss
            training_stat_per_epoch.loss += loss;
            training_stat_per_epoch.total_tags += feat.size() ;
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
        update_epoch(nr_epoch);
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
    return update_recorder.get_record_list();
}


template <typename FeatType, typename NnModel, typename TokenDict>
float devel(NnModel &m, TokenDict &dict,
    const std::pair<std::vector<FeatType>, std::vector<IndexSeq>> &devel_data,
    const std::string& conll_eval_script_path)
{
    unsigned nr_samples = devel_data.size();
    std::cerr << "+ Validation at " << nr_samples << " instances.\n";

    NerStat stat(conlleval_script_path);
    stat.start_time_stat();
    std::vector<FeatType>& feat_list = devel_data.first;
    std::vector<Indexseq>& tag_seq_list = devel_data.second;
    std::vector<IndexSeq> pred_tag_seq_list;
    pred_tag_seq_list.reserve(nr_samples);
    for( unsigned access_idx = 0; access_idx < nr_samples; ++access_idx )
    {
        const FeatType& feat = feat_list[access_idx];
        std::vector<Index> pred_tag_seq = m.predict(feat);
        pred_tag_seq_list.push_back(pred_tag_seq);
        stat.total_tags += pred_tag_seq.size();
    }
    stat.end_time_stat();
    array<float , 4> eval_scores = stat.conlleval(tag_seq_list, pred_tag_seq_list , dict);

    std::ostringstream tmp_sos;
    tmp_sos << "= Validation finished. \n"
        << "| Acc = " << eval_scores[0] << "% , P = " << eval_scores[1]
        << "% , R = " << eval_scores[2] << "% , F1 = " << eval_scores[3] << "%";
    std::cerr << stat.get_stat_str(tmp_sos.str()) << "\n";
    return eval_scores[3];
}


template <typename NnModel, typename TokenDict>
void predict(NnModel &m, TokenDict& dict,
    std::istream &is,
    std::ostream &os)
{
    // TODO!
}


} // end of namespace handler
} // end of namespace ner
} // end of namespace slnn



#endif