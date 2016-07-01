#ifndef SLNN_SEGMENTOR_INPUT1_WITH_FEATURE_MODELHANDLER_0628_H_
#define SLNN_SEGMENTOR_INPUT1_WITH_FEATURE_MODELHANDLER_0628_H_

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "segmentor/base_model/input1_with_feature_model_0628.hpp"
#include "segmentor/cws_module/cws_feature.h"
#include "segmentor/cws_module/cws_tagging_system.h"
#include "utils/stat.hpp"

namespace slnn{

template <typename I1Model>
class CWSInput1WithFeatureModelHandler
{
public :
    CWSInput1WithFeatureModel *i1m ;

    static const std::string OutputDelimiter ;

    CWSInput1WithFeatureModelHandler() ;
    virtual ~CWSInput1WithFeatureModelHandler() ;


    // Reading data 
    void read_training_data(std::istream &is,
        std::vector<IndexSeq> &sents,
        std::vector<IndexSeq> &tag_seqs,
        std::vector<CWSFeatureDataSeq> &feature_data_seq);
    void read_devel_data(std::istream &is,
        std::vector<IndexSeq> &sents,
        std::vector<IndexSeq> &tag_seqs,
        std::vector<CWSFeatureDataSeq> &feature_data_seq);
    virtual void read_test_data(std::istream &is,
        std::vector<Seq> &raw_test_sents, 
        std::vector<IndexSeq> &sents,
        std::vector<CWSFeatureDataSeq> &feature_data_seq);

    // After Reading Training data
    void set_model_param_after_reading_training_data(const boost::program_options::variables_map &varmap);
    void build_model();

    // Train & devel & predict
    void train(const std::vector<IndexSeq> &sents, const std::vector<IndexSeq> &tag_seqs,
        const std::vector<CWSFeatureDataSeq> &feature_data_seqs,
        unsigned max_epoch,
        const std::vector<IndexSeq> &dev_sents, const std::vector<IndexSeq> &dev_tag_seqs,
        const std::vector<CWSFeatureDataSeq> &dev_feature_data_seqs,
        unsigned do_devel_freq,
        unsigned trivial_report_freq);
    float devel(const std::vector<IndexSeq> &sents, const std::vector<IndexSeq> &tag_seqs,
        const std::vector<CWSFeatureDataSeq> &feature_data_seq);
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);
private:
    CNNModelStash model_stash;
};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{

template <typename I1Model>
const std::string CWSInput1WithFeatureModelHandler<I1Model>::OutputDelimiter = "\t" ;

template <typename I1Model>
CWSInput1WithFeatureModelHandler<I1Model>::CWSInput1WithFeatureModelHandler()
    : i1m(new I1Model()) 
{}

template <typename I1Model>
CWSInput1WithFeatureModelHandler<I1Model>::~CWSInput1WithFeatureModelHandler()
{
    delete i1m ;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::read_training_data_and_build_dicts(std::istream &is,
                                        std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs)
{
    assert(!i1m->is_dict_freezon());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    // first , build lexicon
    CWSReader reader(is);
    std::vector<Seq> dataset;
    size_t detected_line_cnt = reader.countline();
    data_set.reserve(detected_line_cnt);
    Seq word_seq;
    while( reader.read_segmented_line(word_seq) )
    {
        i1m->count_word_frequency(word_seq);
        dataset.push_back(std::move(word_seq));
    }
    i1m->build_lexicon();
    // done .

    word_dict_wrapper.Freeze();
    word_dict_wrapper.SetUnk(i1m->UNK_STR);
    tag_dict.Freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set word , tag dict done . ";
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::read_devel_data(std::istream &is,
                                                       std::vector<IndexSeq> &sents, 
                                                       std::vector<IndexSeq> &tag_seqs)
{
    cnn::Dict &word_dict = i1m->get_input_dict() ;
    cnn::Dict &tag_dict = i1m->get_output_dict() ;
    assert(word_dict.is_frozen() && tag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, sents, tag_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::read_test_data(std::istream &is,
                                                      std::vector<Seq> &raw_test_sents, 
                                                      std::vector<IndexSeq> &sents)
{
    cnn::Dict &word_dict = i1m->get_input_dict() ;
    std::string line ;
    std::vector<Seq> tmp_raw_sents ;
    std::vector<IndexSeq> tmp_sents ;

    Seq raw_sent ;
    IndexSeq sent ;
    while( getline(is, line) )
    {
        // do not skip empty line .
        CWSTaggingSystem::split_word(line, raw_sent) ;
        sent.clear() ;
        for( size_t i = 0 ; i < raw_sent.size() ; ++i )
        {
            sent.push_back(word_dict.Convert(raw_sent[i])) ;
        }
        tmp_raw_sents.push_back(raw_sent) ;
        tmp_sents.push_back(sent) ;
    }
    std::swap(raw_test_sents, tmp_raw_sents) ;
    std::swap(sents, tmp_sents) ;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    i1m->set_model_param(varmap) ;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::build_model()
{
    i1m->build_model_structure() ;
    i1m->print_model_info() ;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::train(const std::vector<IndexSeq> *p_sents, 
                                             const std::vector<IndexSeq> *p_tag_seqs,
                                             unsigned max_epoch,
                                             const std::vector<IndexSeq> *p_dev_sents, 
                                             const std::vector<IndexSeq> *p_dev_tag_seqs,
                                             unsigned do_devel_freq,
                                             unsigned trivial_report_freq)
{
    unsigned nr_samples = p_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";
    DictWrapper &word_dict_wrapper = i1m->get_input_dict_wrapper() ;
    std::vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    bool is_train_ok = true ; // when grident update error , we stop the training 
    cnn::SimpleSGDTrainer sgd(i1m->get_cnn_model());
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq dynamic_sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch && is_train_ok; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *cnn::rndeng);

        // For loss , accuracy , time cost report
        BasicStat training_stat_per_epoch;
        training_stat_per_epoch.start_time_stat();

        // train for every Epoch 
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq &sent = p_sents->at(access_idx),
                &tag_seq = p_tag_seqs->at(access_idx);
            { // new scope , for only one Computatoin Graph can be exists in one scope at the same time .
              // devel will creat another Computation Graph , so we need to create new scoce to release it before devel .
                cnn::ComputationGraph cg ;
                dynamic_sent_after_replace_unk.resize(sent.size());
                for( size_t word_idx = 0; word_idx < sent.size(); ++word_idx )
                {
                    dynamic_sent_after_replace_unk[word_idx] =
                        word_dict_wrapper.ConvertProbability(sent.at(word_idx));
                }
                i1m->build_loss(cg, dynamic_sent_after_replace_unk, tag_seq);
                cnn::real loss = as_scalar(cg.forward());
                cg.backward();
                sgd.update(1.f);
                training_stat_per_epoch.loss += loss;
                training_stat_per_epoch.total_tags += sent.size() ;
            }
            if (0 == (i + 1) % trivial_report_freq) // Report 
            {
                std::string trivial_header = std::to_string(i + 1) + " instances have been trained.";
                BOOST_LOG_TRIVIAL(trace) << training_stat_per_epoch.get_stat_str(trivial_header);
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if (p_dev_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
            {
                float F1 = devel(p_dev_sents  , p_dev_tag_seqs);
                if (F1 > best_F1) save_current_best_model(F1);
                line_cnt_for_devel = 0; // avoid overflow
                if( is_train_error_occurs(F1) )
                {
                    is_train_ok = false;
                    break ;
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
        if (p_dev_sents != nullptr && is_train_ok)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float F1 = devel(p_dev_sents , p_dev_tag_seqs);
            if (F1 > best_F1) save_current_best_model(F1);
            if( is_train_error_occurs(F1) )
            {
                is_train_ok = false ;
                break ;
            }
        }

    }
    if( !is_train_ok ){ BOOST_LOG_TRIVIAL(warning) << "Gradient may have been updated error ! Exit ahead of time." ; }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

template <typename I1Model>
float CWSInput1WithFeatureModelHandler<I1Model>::devel(const std::vector<IndexSeq> *p_sents, 
                                              const std::vector<IndexSeq> *p_tag_seqs)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    CWSStat stat(i1m->get_tag_sys() , true);
    stat.start_time_stat();
    std::vector<IndexSeq> predict_tag_seqs(p_tag_seqs->size());
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        cnn::ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq &sent = p_sents->at(access_idx);
        i1m->predict(cg, sent,predict_tag_seq);
        predict_tag_seqs[access_idx] = predict_tag_seq;
        stat.total_tags += predict_tag_seq.size();
    }
    stat.end_time_stat();
    std::array<float , 4> eval_scores = stat.eval(*p_tag_seqs, predict_tag_seqs);
    float Acc = eval_scores[0] , 
        P = eval_scores[1] ,
        R = eval_scores[2] ,
        F1 = eval_scores[3] ; // they are all PERCENT VALUE
    std::ostringstream tmp_sos;
    tmp_sos << "validation finished .\n"
        << "Acc = " << Acc << "% , P = " << P << "% , R = " << R << "% , F1 = " << F1 << "%";
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str(tmp_sos.str()) ;
    return F1;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::predict(std::istream &is, std::ostream &os)
{
    
    std::vector<Seq> raw_instances;
    std::vector<IndexSeq> sents ;
    read_test_data(is,raw_instances, sents );
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
        IndexSeq pred_tag_seq;
        cnn::ComputationGraph cg;
        i1m->predict(cg, sent, pred_tag_seq);
        Seq words ;
        i1m->get_tag_sys().parse_word_tag2words(raw_sent, pred_tag_seq, words) ;
        os << words[0] ;
        for( size_t i = 1 ; i < words.size() ; ++i ) os << OUT_SPLIT_DELIMITER << words[i] ;
        os << "\n";
        stat.total_tags += pred_tag_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    if( best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail() )
    {
        BOOST_LOG_TRIVIAL(info) << "fetch best model ...";
        i1m->set_cnn_model(best_model_tmp_ss) ;
    }
    boost::archive::text_oarchive to(os);
    to << *(static_cast<I1Model*>(i1m));
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

template <typename I1Model>
void CWSInput1WithFeatureModelHandler<I1Model>::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is) ;
    ti >> *(static_cast<I1Model*>(i1m));
    i1m->print_model_info() ;
}

} // end of namespace slnn
#endif
