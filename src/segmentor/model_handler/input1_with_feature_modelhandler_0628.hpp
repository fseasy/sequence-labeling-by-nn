#ifndef SLNN_SEGMENTOR_INPUT1_WITH_FEATURE_MODELHANDLER_0628_H_
#define SLNN_SEGMENTOR_INPUT1_WITH_FEATURE_MODELHANDLER_0628_H_

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "segmentor/base_model/input1_with_feature_model_0628.hpp"
#include "segmentor/cws_module/cws_feature.h"
#include "segmentor/cws_module/cws_tagging_system.h"
#include "utils/stat.hpp"
#include "utils/stash_model.hpp"
#include "segmentor/cws_module/cws_reader.h"
namespace slnn{

template <typename RNNDerived, typename I1Model>
class CWSInput1WithFeatureModelHandler
{
public :
    CWSInput1WithFeatureModel<RNNDerived> *i1m ;

    static const std::string OutputDelimiter ;

    CWSInput1WithFeatureModelHandler() ;
    virtual ~CWSInput1WithFeatureModelHandler() ;


    // Reading data 
    void read_training_data(std::istream &is,
        std::vector<IndexSeq> &sents,
        std::vector<CWSFeatureDataSeq> &feature_data_seq,
        std::vector<IndexSeq> &tag_seqs);
    void read_devel_data(std::istream &is,
        std::vector<IndexSeq> &sents,
        std::vector<CWSFeatureDataSeq> &feature_data_seq,
        std::vector<IndexSeq> &tag_seqs);
    void read_test_data(std::istream &is,
        std::vector<Seq> &raw_test_sents, 
        std::vector<IndexSeq> &sents,
        std::vector<CWSFeatureDataSeq> &feature_data_seq);

    // After Reading Training data
    void set_model_param_after_reading_training_data(const boost::program_options::variables_map &varmap);
    void build_model();

    // Train & devel & predict
    void train(const std::vector<IndexSeq> &sents, 
        const std::vector<CWSFeatureDataSeq> &feature_data_seqs,
        const std::vector<IndexSeq> &tag_seqs,
        unsigned max_epoch,
        const std::vector<IndexSeq> &dev_sents, 
        const std::vector<CWSFeatureDataSeq> &dev_feature_data_seqs,
        const std::vector<IndexSeq> &dev_tag_seqs,
        unsigned do_devel_freq,
        unsigned trivial_report_freq);
    float devel(const std::vector<IndexSeq> &sents, 
        const std::vector<CWSFeatureDataSeq> &feature_data_seq,
        const std::vector<IndexSeq> &tag_seqs);
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);
private:
    CNNModelStash model_stash;
};

} // end of namespace slnn

/**********************************************
    implementation
***********************************************/

namespace slnn{

template <typename RNNDerived, typename I1Model>
const std::string CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::OutputDelimiter = "\t" ;

template <typename RNNDerived, typename I1Model>
CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::CWSInput1WithFeatureModelHandler()
    : i1m(new I1Model()) 
{}

template <typename RNNDerived, typename I1Model>
CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::~CWSInput1WithFeatureModelHandler()
{
    delete i1m ;
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::read_training_data(std::istream &is,
    std::vector<IndexSeq> &sents,
    std::vector<CWSFeatureDataSeq> &cws_feature_seqs,
    std::vector<IndexSeq> &tag_seqs)
{
    using std::swap;
    assert(!i1m->is_dict_frozen());
    // first , build lexicon
    BOOST_LOG_TRIVIAL(info) << "+ build lexicon from training data.";
    CWSReader reader(is);
    std::vector<Seq> dataset;
    size_t detected_line_cnt = reader.count_line();
    dataset.reserve(detected_line_cnt);
    Seq word_seq;
    while( reader.read_segmented_line(word_seq) )
    {
        if( word_seq.empty() ){ continue; }
        i1m->count_word_frequency(word_seq);
        dataset.push_back(std::move(word_seq));
    }
    i1m->build_lexicon();
    // translate word str to char index , extract feature
    BOOST_LOG_TRIVIAL(info) << "+ process training data.";
    unsigned dataset_size = dataset.size();
    std::vector<IndexSeq> tmp_sents(dataset_size),
        tmp_tag_seqs(dataset_size);
    std::vector<CWSFeatureDataSeq> tmp_cws_feature_seqs(dataset_size);
    for( size_t i = 0; i < dataset_size; ++i )
    {
        i1m->word_seq2index_seq(dataset[i], tmp_sents[i], tmp_tag_seqs[i], tmp_cws_feature_seqs[i]);
        if( (i+1) % 10000 == 0 ){ BOOST_LOG_TRIVIAL(info) << i+1 << " instances has been processed." ; }
    }
    i1m->freeze_dict();
    BOOST_LOG_TRIVIAL(info) << "- Training data processed done. totally " << dataset_size << " instances has been processed.";
    swap(sents, tmp_sents);
    swap(tag_seqs, tmp_tag_seqs);
    swap(cws_feature_seqs, tmp_cws_feature_seqs);
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::read_devel_data(std::istream &is,
    std::vector<IndexSeq> &sents,
    std::vector<CWSFeatureDataSeq> &cws_feature_seqs,
    std::vector<IndexSeq> &tag_seqs)
{
    using std::swap;
    assert(i1m->is_dict_frozen());
    BOOST_LOG_TRIVIAL(info) << "+ process devel data .";
    CWSReader reader(is);
    size_t detected_line_cnt = reader.count_line();
    std::vector<IndexSeq> tmp_sents,
        tmp_tag_seqs;
    std::vector<CWSFeatureDataSeq> tmp_cws_feature_seqs;
    tmp_sents.reserve(detected_line_cnt);
    tmp_tag_seqs.reserve(detected_line_cnt);
    tmp_cws_feature_seqs.reserve(detected_line_cnt);
    size_t line_cnt = 0;
    Seq word_seq;
    while( reader.read_segmented_line(word_seq) )
    {
        IndexSeq char_seq, tag_seq;
        CWSFeatureDataSeq cws_feature_seq;
        i1m->word_seq2index_seq(word_seq, char_seq, tag_seq, cws_feature_seq);
        tmp_sents.push_back(std::move(char_seq));
        tmp_tag_seqs.push_back(std::move(tag_seq));
        tmp_cws_feature_seqs.push_back(std::move(cws_feature_seq));
        ++line_cnt;
        if( line_cnt % 10000 == 0 ){ BOOST_LOG_TRIVIAL(info) << line_cnt << " instances has been processed."; }
    }
    BOOST_LOG_TRIVIAL(info) << "- Devel data processed done. totally " << line_cnt << " instances has been processed.";
    swap(sents, tmp_sents);
    swap(tag_seqs, tmp_tag_seqs);
    swap(cws_feature_seqs, tmp_cws_feature_seqs);
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::read_test_data(std::istream &is,
    std::vector<Seq> &raw_test_sents,
    std::vector<IndexSeq> &sents,
    std::vector<CWSFeatureDataSeq> &cws_feature_seqs)
{
    using std::swap;
    assert(i1m->is_dict_frozen());
    BOOST_LOG_TRIVIAL(info) << "+ processing test data.";
    CWSReader reader(is);
    size_t detected_line_cnt = reader.count_line();
    std::vector<Seq> tmp_raw_test_sents;
    std::vector<IndexSeq> tmp_sents;
    std::vector<CWSFeatureDataSeq> tmp_cws_feature_seqs;
    tmp_raw_test_sents.reserve(detected_line_cnt);
    tmp_sents.reserve(detected_line_cnt);
    tmp_cws_feature_seqs.reserve(detected_line_cnt);

    size_t line_cnt = 0;
    Seq char_seq;
    while( reader.readline(char_seq) )
    {
        IndexSeq sent;
        CWSFeatureDataSeq feature_seq;
        i1m->char_seq2index_seq(char_seq, sent, feature_seq);
        tmp_raw_test_sents.push_back(std::move(char_seq));
        tmp_sents.push_back(std::move(sent));
        tmp_cws_feature_seqs.push_back(std::move(feature_seq));

        if( ++line_cnt % 10000 == 0 ){ BOOST_LOG_TRIVIAL(info) << line_cnt << " instances has been processed."; }
    }
    BOOST_LOG_TRIVIAL(info) << "- Test data processed done. totally " << line_cnt << " instances has been processed.";
    swap(raw_test_sents, tmp_raw_test_sents);
    swap(sents, tmp_sents);
    swap(cws_feature_seqs, tmp_cws_feature_seqs);
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::
set_model_param_after_reading_training_data(const boost::program_options::variables_map &varmap)
{
    i1m->set_model_param(varmap) ;
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::build_model()
{
    i1m->build_model_structure() ;
    i1m->print_model_info() ;
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::train(const std::vector<IndexSeq> &sents,
    const std::vector<CWSFeatureDataSeq> &cws_feature_seqs,
    const std::vector<IndexSeq> &tag_seqs,
    unsigned max_epoch,
    const std::vector<IndexSeq> &dev_sents,
    const std::vector<CWSFeatureDataSeq> &dev_cws_feature_seqs,
    const std::vector<IndexSeq> &dev_tag_seqs,
    unsigned do_devel_freq,
    unsigned trivial_report_freq)
{
    unsigned nr_samples = sents.size();

    BOOST_LOG_TRIVIAL(info) << "+ Train at " << nr_samples << " instances .";
    std::vector<unsigned> access_order(nr_samples);
    for( unsigned i = 0; i < nr_samples; ++i ) access_order[i] = i;

    cnn::SimpleSGDTrainer sgd(i1m->get_cnn_model());

    auto do_devel_in_training = [this, &dev_sents, &dev_cws_feature_seqs, &dev_tag_seqs](CNNModelStash &model_stash) 
    {
        // CNNModelStash as param to remind we'll change it's state !
        float F1 = this->devel(dev_sents, dev_cws_feature_seqs, dev_tag_seqs);
        model_stash.save_when_best(this->i1m->get_cnn_model(), F1);
        model_stash.update_training_state(F1);
    };
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    for( unsigned nr_epoch = 0; nr_epoch < max_epoch ; ++nr_epoch )
    {
        BOOST_LOG_TRIVIAL(info) << "++ Epoch " << nr_epoch + 1 << "/" << max_epoch << " start ";
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
            const IndexSeq &sent = sents.at(access_idx),
                &tag_seq = tag_seqs.at(access_idx);
            const CWSFeatureDataSeq &cws_feature_seq = cws_feature_seqs.at(access_idx);
            { // new scope , for only one Computatoin Graph can be exists in one scope at the same time .
              // devel will creat another Computation Graph , so we need to create new scoce to release it before devel .
                cnn::ComputationGraph cg ;
                IndexSeq replaced_sent;
                i1m->replace_word_with_unk(sent, replaced_sent);
                i1m->build_loss(cg, replaced_sent, cws_feature_seq, tag_seq);
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

            ++line_cnt_for_devel;
            // do devel at every `do_devel_freq`
            if( 0 == line_cnt_for_devel % do_devel_freq )
            {
                do_devel_in_training(model_stash);
                if( !model_stash.is_training_ok() ){ break;  }
            }
        }

        // End of an epoch 
        sgd.update_epoch();

        training_stat_per_epoch.end_time_stat();
        // Output at end of every eopch
        std::ostringstream tmp_sos;
        tmp_sos << "- Epoch " << nr_epoch + 1 << "/" << std::to_string(max_epoch) << " finished .\n"
            << nr_samples << " instances has been trained . ";
        std::string info_header = tmp_sos.str();
        BOOST_LOG_TRIVIAL(info) << training_stat_per_epoch.get_stat_str(info_header);
        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
        // do validation at every ends of epoch
        if( model_stash.is_training_ok() )
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            do_devel_in_training(model_stash);
        }
        if( !model_stash.is_training_ok() ){ break; }
    }
    if( !model_stash.is_training_ok() ){ BOOST_LOG_TRIVIAL(warning) << "Gradient may have been updated error ! Exit ahead of time." ; }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

template <typename RNNDerived, typename I1Model>
float CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::devel(const std::vector<IndexSeq> &sents,
    const std::vector<CWSFeatureDataSeq> &cws_feature_seqs,
    const std::vector<IndexSeq> &tag_seqs)
{
    unsigned nr_samples = sents.size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    CWSStatNew stat(true);
    stat.start_time_stat();
    std::vector<IndexSeq> predict_tag_seqs(tag_seqs.size());
    for( unsigned access_idx = 0; access_idx < nr_samples; ++access_idx )
    {
        cnn::ComputationGraph cg;
        const IndexSeq &sent = sents.at(access_idx);
        const CWSFeatureDataSeq &feature_seq = cws_feature_seqs.at(access_idx);
        i1m->predict(cg, sent, feature_seq, predict_tag_seqs[access_idx]);
        stat.total_tags += predict_tag_seqs[access_idx].size();
    }
    stat.end_time_stat();
    std::array<float, 4> eval_scores = stat.eval(tag_seqs, predict_tag_seqs);
    float Acc = eval_scores[0],
        P = eval_scores[1],
        R = eval_scores[2],
        F1 = eval_scores[3] ; // they are all PERCENT VALUE
    std::ostringstream tmp_sos;
    tmp_sos << "validation finished .\n"
        << "Acc = " << Acc << "% , P = " << P << "% , R = " << R << "% , F1 = " << F1 << "%";
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str(tmp_sos.str()) ;
    return F1;
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::predict(std::istream &is, std::ostream &os)
{
    
    std::vector<Seq> raw_instances;
    std::vector<IndexSeq> sents ;
    std::vector<CWSFeatureDataSeq> cws_feature_seqs;
    read_test_data(is, raw_instances, sents, cws_feature_seqs );
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
        CWSFeatureDataSeq &cws_feature_seq = cws_feature_seqs.at(i);
        IndexSeq pred_tag_seq;
        cnn::ComputationGraph cg;
        i1m->predict(cg, sent, cws_feature_seq, pred_tag_seq);
        Seq words ;
        CWSTaggingSystem::static_parse_chars_indextag2word_seq(raw_sent, pred_tag_seq, words) ;
        os << words[0] ;
        for( size_t i = 1 ; i < words.size() ; ++i ) os << OutputDelimiter << words[i] ;
        os << "\n";
        stat.total_tags += pred_tag_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    model_stash.load_if_exists(i1m->get_cnn_model());
    boost::archive::text_oarchive to(os);
    to << *(static_cast<I1Model*>(i1m));
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

template <typename RNNDerived, typename I1Model>
void CWSInput1WithFeatureModelHandler<RNNDerived, I1Model>::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is) ;
    ti >> *(static_cast<I1Model*>(i1m));
    i1m->print_model_info() ;
}

} // end of namespace slnn
#endif
