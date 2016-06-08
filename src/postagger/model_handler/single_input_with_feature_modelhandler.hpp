#ifndef POS_MODEL_HANDLER_SINGLE_INPUT_WITH_FEATURE_MODELHANDLERS_HPP_
#define POS_MODEL_HANDLER_SINGLE_INPUT_WITH_FEATURE_MODELHANDLERS_HPP_
#include <iostream>
#include <vector>
#include "postagger/base_model/single_input_with_feature_model.hpp"
#include "postagger/postagger_module/pos_reader.h"
#include "utils/stash_model.hpp"
#include "utils/stat.hpp"
namespace slnn{

template <typename RNNDerived, typename SIModel>
class SingleInputWithFeatureModelHandler
{
public :
    SingleInputModelWithFeature<RNNDerived> *sim;

    static const size_t MaxSentNum = 0x8000; // 32k
    static const std::string OutputDelimiter ;

    SingleInputWithFeatureModelHandler();
    ~SingleInputWithFeatureModelHandler();
    SingleInputWithFeatureModelHandler(const SingleInputWithFeatureModelHandler&) = delete;
    SingleInputWithFeatureModelHandler& operator()(const SingleInputWithFeatureModelHandler&) = delete;


    // Reading data 
    void read_annotated_data(std::istream &is,
                             std::vector<IndexSeq> &sents,
                             std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_seqs,
                             std::vector<IndexSeq> &postags_seqs);

    void read_training_data(std::istream &is, 
                            std::vector<IndexSeq> &training_sents,
                            std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_seqs,
                            std::vector<IndexSeq> &postags_seqs);

    void read_devel_data(std::istream &is,
                         std::vector<IndexSeq> &devel_sents,
                         std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_seqs,
                         std::vector<IndexSeq> &postag_seqs);
    
    void read_test_data(std::istream &is,
                        std::vector<Seq> &raw_sents,
                        std::vector<IndexSeq> &sents,
                        std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_seqs);

    void train(const std::vector<IndexSeq> *p_sents,
               const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_feature_gp_seqs,
               const std::vector<IndexSeq> *p_tag_seqs,
               unsigned max_epoch,
               const std::vector<IndexSeq> *p_dev_sents,
               const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_dev_features_gp_seqs,
               const std::vector<IndexSeq> *p_dev_tag_seqs,
               unsigned do_devel_freq,
               unsigned trivial_report_freq);

    float devel(const std::vector<IndexSeq> *p_sents,
                const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_feature_gp_seqs,
                const std::vector<IndexSeq> *p_tag_seqs);

    void predict(std::istream &is, std::ostream &os);

    void save_model(std::ostream &os);
    void load_model(std::istream &is);

    // After read data
    void set_model_param_after_reading_training_data(const boost::program_options::variables_map &varmap);
    void build_model();

private :
    CNNModelStash model_stash;
};

template<typename RNNDerived, typename SIModel>
const size_t SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::MaxSentNum;

template <typename RNNDerived, typename SIModel>
const std::string SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::OutputDelimiter = "\t";

template <typename RNNDerived, typename SIModel>
SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::SingleInputWithFeatureModelHandler()
    :sim(new SIModel())
{}

template <typename RNNDerived, typename SIModel>
SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::~SingleInputWithFeatureModelHandler()
{
    delete sim;
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::read_annotated_data(std::istream &is,
                         std::vector<IndexSeq> &sents,
                         std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_gp_seqs,
                         std::vector<IndexSeq> &postag_seqs)
{
    using std::swap;
    POSReader reader(is);
    std::vector<IndexSeq> tmp_sents,
        tmp_postag_seqs;
    std::vector<POSFeature::POSFeatureIndexGroupSeq> tmp_features_gp_seqs;
    size_t detected_line_cnt = reader.count_line();
    tmp_sents.reserve(detected_line_cnt);
    tmp_postag_seqs.reserve(detected_line_cnt);
    tmp_features_gp_seqs.reserve(detected_line_cnt);
    size_t line_cnt = 0 ;
    while( reader.good() )
    {
        Seq str_sent,
            str_postag_seq;
        reader.readline(str_sent, str_postag_seq);
        if( str_sent.size() == 0 ) continue;
        IndexSeq sent,
            postag_seq;
        POSFeature::POSFeatureIndexGroupSeq features_gp_seq;
        sim->input_seq2index_seq(str_sent, str_postag_seq, sent, postag_seq, features_gp_seq);
        tmp_sents.push_back(std::move(sent));
        tmp_postag_seqs.push_back(std::move(postag_seq));
        tmp_features_gp_seqs.push_back(std::move(features_gp_seq));
        ++line_cnt;
        if( line_cnt % 10000 ) BOOST_LOG_TRIVIAL(info) << line_cnt << " lines has been preprocessed."  ;
    }
    swap(sents, tmp_sents);
    swap(features_gp_seqs, tmp_features_gp_seqs);
    swap(postag_seqs, tmp_postag_seqs);
    BOOST_LOG_TRIVIAL(info) << "Totally " << line_cnt << " lines has been preprocessed done." ;
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::read_training_data(std::istream &is,
                                                                                 std::vector<IndexSeq> &training_sents,
                                                                                 std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_gp_seqs,
                                                                                 std::vector<IndexSeq> &postag_seqs)
{
    assert(!sim->is_dict_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    read_annotated_data(is, training_sents, features_gp_seqs, postag_seqs);
    sim->freeze_dict();
    BOOST_LOG_TRIVIAL(info) << "read training data done.";
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::read_devel_data(std::istream &is,
                                                                              std::vector<IndexSeq> &devel_sents,
                                                                              std::vector<POSFeature::POSFeatureIndexGroupSeq> &features_gp_seqs,
                                                                              std::vector<IndexSeq> &postag_seqs)
{
    assert(sim->is_dict_frozen());
    BOOST_LOG_TRIVIAL(info) << "read devel data .";
    read_annotated_data(is, devel_sents, features_gp_seqs, postag_seqs);
    BOOST_LOG_TRIVIAL(info) << "read devel data done.";
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::read_test_data(std::istream &is,
                                                                             std::vector<Seq> &raw_sents,
                                                                             std::vector<IndexSeq> &sents,
                                                                             std::vector<POSFeature::POSFeatureIndexGroupSeq> &feature_gp_seqs)
{
    using std::swap;
    assert(sim->is_dict_frozen());
    BOOST_LOG_TRIVIAL(info) << "read data." ;
    std::vector<Seq> tmp_raw_sents;
    std::vector<IndexSeq> tmp_sents;
    std::vector<POSFeature::POSFeatureIndexGroupSeq> tmp_feature_gp_seqs;
    POSReader reader(is);
    size_t detected_line_cnt = reader.count_line();
    tmp_raw_sents.reserve(detected_line_cnt);
    tmp_sents.reserve(detected_line_cnt);
    tmp_feature_gp_seqs.reserve(detected_line_cnt);
    size_t line_cnt = 0 ;
    while( reader.good() )
    {
        Seq raw_sent;
        reader.readline(raw_sent);
        IndexSeq sent;
        POSFeature::POSFeatureIndexGroupSeq feature_gp_seq;
        sim->input_seq2index_seq(raw_sent, sent, feature_gp_seq);
        tmp_raw_sents.push_back(std::move(raw_sent));
        tmp_sents.push_back(std::move(sent));
        tmp_feature_gp_seqs.push_back(std::move(feature_gp_seq));
        ++line_cnt;
        if( 0 == line_cnt % 10000 ){ BOOST_LOG_TRIVIAL(info) << line_cnt << " lines has been preprocessed."; }
    }
    swap(raw_sents, tmp_raw_sents);
    swap(sents, tmp_sents);
    swap(feature_gp_seqs, tmp_feature_gp_seqs);
    BOOST_LOG_TRIVIAL(info) << line_cnt << " lines has been preprocessed done.";
    BOOST_LOG_TRIVIAL(info) << "read data done." ;
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::
set_model_param_after_reading_training_data(const boost::program_options::variables_map &varmap)
{
    sim->set_model_param(varmap);
}

template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::build_model()
{
    sim->build_model_structure();
    sim->print_model_info();
}


template <typename RNNDerived, typename SIModel>
void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::train(const std::vector<IndexSeq> *p_sents,
                                                                    const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_feature_gp_seqs,
                                                                    const std::vector<IndexSeq> *p_tag_seqs,
                                                                    unsigned max_epoch,
                                                                    const std::vector<IndexSeq> *p_dev_sents,
                                                                    const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_dev_feature_gp_seqs,
                                                                    const std::vector<IndexSeq> *p_dev_tag_seqs,
                                                                    unsigned do_devel_freq,
                                                                    unsigned trivial_report_freq)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";

    std::vector<unsigned> access_order(nr_samples);
    for( unsigned i = 0; i < nr_samples; ++i ) access_order[i] = i;

    bool is_train_ok = true;
    cnn::SimpleSGDTrainer sgd(sim->get_cnn_model());
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    for( unsigned nr_epoch = 0; nr_epoch < max_epoch && is_train_ok; ++nr_epoch )
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *cnn::rndeng);

        // For loss , accuracy , time cost report
        BasicStat training_stat_per_epoch;
        training_stat_per_epoch.start_time_stat();

        // train for every Epoch 
        for( unsigned i = 0; i < nr_samples; ++i )
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq &sent = p_sents->at(access_idx),
                &tag_seq = p_tag_seqs->at(access_idx);
            const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq = p_feature_gp_seqs->at(access_idx);
            { // new scope , for only one Computatoin Graph can be exists in one scope at the same time .
              // devel will creat another Computation Graph , so we need to create new scoce to release it before devel .
                cnn::ComputationGraph cg ;
                IndexSeq sent_after_replace ;
                POSFeature::POSFeatureIndexGroupSeq feature_gp_seq_after_replace;
                sim->replace_word_with_unk(sent, feature_gp_seq, sent_after_replace, feature_gp_seq_after_replace);
                sim->build_loss(cg, sent_after_replace, feature_gp_seq_after_replace, tag_seq);
                cnn::real loss = as_scalar(cg.forward());
                cg.backward();
                sgd.update(1.f);
                training_stat_per_epoch.loss += loss;
                training_stat_per_epoch.total_tags += sent.size() ;
            }
            if( 0 == (i + 1) % trivial_report_freq ) // Report 
            {
                std::string trivial_header = std::to_string(i + 1) + " instances have been trained.";
                BOOST_LOG_TRIVIAL(trace) << training_stat_per_epoch.get_stat_str(trivial_header);
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if( p_dev_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq )
            {
                float acc = devel(p_dev_sents, p_dev_feature_gp_seqs, p_dev_tag_seqs);
                model_stash.save_when_best(sim->get_cnn_model(), acc);
                line_cnt_for_devel = 0; // avoid overflow
                if( model_stash.is_train_error_occurs(acc) )
                {
                    is_train_ok = false;
                    break;
                }
            }
        }

        // End of an epoch 
        sgd.update_epoch();

        training_stat_per_epoch.end_time_stat();
        // Output at end of every eopch
        std::ostringstream tmp_sos;
        tmp_sos << "-------- epoch " << nr_epoch + 1 << "/" << std::to_string(max_epoch) << " finished . ----------\n"
            << nr_samples << " instances has been trained . ";
        std::string info_header = tmp_sos.str();
        BOOST_LOG_TRIVIAL(info) << training_stat_per_epoch.get_stat_str(info_header);
        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
        // do validation at every ends of epoch
        if( p_dev_sents != nullptr && is_train_ok )
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float acc = devel(p_dev_sents, p_dev_feature_gp_seqs, p_dev_tag_seqs);
            model_stash.save_when_best(sim->get_cnn_model(), acc);
            if( model_stash.is_train_error_occurs(acc) )
            {
                is_train_ok = false;
                break;
            }
        }

    }
    if( !is_train_ok ){ BOOST_LOG_TRIVIAL(warning) << "Gradient may have been updated error ! Exit ahead of time." ; }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}


                                                                    template <typename RNNDerived, typename SIModel>
                                                                    float SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::devel(const std::vector<IndexSeq> *p_sents, 
                                                                                                                                         const std::vector<POSFeature::POSFeatureIndexGroupSeq> *p_feature_gp_seqs,
                                              const std::vector<IndexSeq> *p_tag_seqs)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    Stat stat(true);
    stat.start_time_stat();
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        cnn::ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq &sent = p_sents->at(access_idx) ,
            &gold_tag = p_tag_seqs->at(access_idx);
        const POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq = p_feature_gp_seqs->at(access_idx);
        sim->predict(cg, sent,feature_gp_seq, predict_tag_seq);

        stat.total_tags += predict_tag_seq.size();
        for( size_t tag_idx = 0 ; tag_idx <= gold_tag.size() ; ++tag_idx )
        {
            if( gold_tag[tag_idx] == predict_tag_seq[tag_idx] ) ++stat.correct_tags ;
        }
    }
    stat.end_time_stat();
    
    std::ostringstream tmp_sos;
    tmp_sos << "validation finished ." ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str(tmp_sos.str()) ;
    return stat.get_acc() ;
}


                                                                    template <typename RNNDerived, typename SIModel>
                                                                    void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::predict(std::istream &is, std::ostream &os)
{
    
    std::vector<Seq> raw_instances;
    std::vector<IndexSeq> sents ;
    std::vector<POSFeature::POSFeatureIndexGroupSeq> feature_gp_seqs;
    read_test_data(is,raw_instances, sents, feature_gp_seqs );
    BOOST_LOG_TRIVIAL(info) << "do prediction on " << raw_instances.size() << " instances .";
    BasicStat stat(true);
    stat.start_time_stat();
    for (unsigned int i = 0; i < raw_instances.size(); ++i)
    {
        Seq &raw_sent = raw_instances.at(i);
        if (0 == raw_sent.size())
        {
            os << "\n";
            continue;
        }
        IndexSeq &sent = sents.at(i) ;
        POSFeature::POSFeatureIndexGroupSeq &feature_gp_seq = feature_gp_seqs.at(i);
        IndexSeq pred_tag_seq;
        cnn::ComputationGraph cg;
        sim->predict(cg, sent, feature_gp_seq, pred_tag_seq);
        Seq postag_seq;
        sim->postag_index_seq2postag_str_seq(pred_tag_seq, postag_seq);
        os << raw_sent[0] << "_" << postag_seq[0] ;
        for( size_t i = 1 ; i < raw_sent.size() ; ++i )
        { 
            os << OutputDelimiter
                << raw_sent[i] << "_" << postag_seq[i] ; 
        }
        os << "\n";
        stat.total_tags += pred_tag_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

                                                                    template <typename RNNDerived, typename SIModel>
                                                                    void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    model_stash.load_if_exists(sim->get_cnn_model());
    boost::archive::text_oarchive to(os);
    to << *(static_cast<SIModel*>(sim));
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

                                                                    template <typename RNNDerived, typename SIModel>
                                                                    void SingleInputWithFeatureModelHandler<RNNDerived, SIModel>::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is) ;
    ti >> *(static_cast<SIModel*>(sim));
    sim->print_model_info() ;
}

} // end of namespace slnn


#endif