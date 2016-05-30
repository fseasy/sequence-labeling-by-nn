#ifndef SLNN_SEGMENTOR_CWS_SINGLE_CLASSIFICATION_SINGLE_INPUT_MODELHANDLER_H
#define SLNN_SEGMENTOR_CWS_SINGLE_CLASSIFICATION_SINGLE_INPUT_MODELHANDLER_H

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "segmentor/base_model/single_input_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"

#include "utils/stat.hpp"
namespace slnn{

template <typename SIModel>
class SingleInputModelHandler
{
public :
    SingleInputModel *sim ;

    float best_F1;
    std::stringstream best_model_tmp_ss;

    const size_t SentMaxLen = 256;
    const size_t MaxSentNum = 0x8000; // 32k
    static const std::string OUT_SPLIT_DELIMITER ;

    SingleInputModelHandler() ;
    ~SingleInputModelHandler() ;
    // Before read data
    void set_unk_replace_threshold(int freq_thres , float prob_thres);

    // Reading data 
    void do_read_annotated_dataset(std::istream &is, 
                                   std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs);
    void read_training_data_and_build_dicts(std::istream &is, 
                                            std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs);
    void read_devel_data(std::istream &is, 
                         std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs);
    void read_test_data(std::istream &is,
                        std::vector<Seq> &raw_test_sents, std::vector<IndexSeq> &sents);

    // After Reading Training data
    void finish_read_training_data(boost::program_options::variables_map &varmap);
    void build_model();

    // Train & devel & predict
    void train(const std::vector<IndexSeq> *p_sents, const std::vector<IndexSeq> *p_tag_seqs ,
               unsigned max_epoch, 
               const std::vector<IndexSeq> *p_dev_sents, const std::vector<IndexSeq> *p_dev_tag_seqs ,
               unsigned do_devel_freq ,
               unsigned trivial_report_freq);
    float devel(const std::vector<IndexSeq> *p_sents, const std::vector<IndexSeq> *p_tag_seqs );
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);
private :
    inline void save_current_best_model(float F1);

};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{
template <typename SIModel>
const std::string SingleInputModelHandler<SIModel>::OUT_SPLIT_DELIMITER = "\t" ;

template <typename SIModel>
SingleInputModelHandler<SIModel>::SingleInputModelHandler()
    : sim(new SIModel()) ,
    best_F1(0.f) ,
    best_model_tmp_ss()
{}

template <typename SIModel>
SingleInputModelHandler<SIModel>::~SingleInputModelHandler()
{
    delete sim ;
}

template<typename SIModel>
inline 
void SingleInputModelHandler<SIModel>::save_current_best_model(float F1)
{
    BOOST_LOG_TRIVIAL(info) << "better model has been found . stash it .";
    best_F1 = F1;
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss);
    to << *sim->get_cnn_model();
}

template<typename SIModel>
void SingleInputModelHandler<SIModel>::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    sim->get_input_dict_wrapper().set_threshold(freq_thres, prob_thres);
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::do_read_annotated_dataset(std::istream &is,
                    std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs)
{
    unsigned line_cnt = 0;
    std::string line;
    std::vector<IndexSeq> tmp_sents,
        tmp_tag_seqs;
    IndexSeq sent, 
        tag_seq;
    // pre-allocation
    tmp_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_tag_seqs.reserve(MaxSentNum);

    sent.reserve(SentMaxLen);
    tag_seq.reserve(SentMaxLen);

    DictWrapper &word_dict_wrapper = sim->get_input_dict_wrapper() ;
    cnn::Dict &tag_dict = sim->get_output_dict() ;
    while (getline(is, line)) {
        if (0 == line.size()) continue;
        sent.clear() ;
        tag_seq.clear() ;
        std::istringstream iss(line) ;
        std::string words_line ;
        Seq tmp_word_cont,
            tmp_tag_cont ;
        while( iss >> words_line )
        {
            CWSTaggingSystem::parse_words2word_tag(words_line, tmp_word_cont, tmp_tag_cont) ;
            for( size_t i = 0 ; i < tmp_word_cont.size() ; ++i )
            {
                Index word_id = word_dict_wrapper.Convert(tmp_word_cont[i]) ;
                Index tag_id = tag_dict.Convert(tmp_tag_cont[i]) ;
                sent.push_back(word_id) ;
                tag_seq.push_back(tag_id) ;
            }
        }
        tmp_sents.push_back(sent);
        tmp_tag_seqs.push_back(tag_seq);
        ++line_cnt;
        if(0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    std::swap(sents, tmp_sents);
    std::swap(tag_seqs, tmp_tag_seqs);
}
template <typename SIModel>
void SingleInputModelHandler<SIModel>::read_training_data_and_build_dicts(std::istream &is,
                                        std::vector<IndexSeq> &sents, std::vector<IndexSeq> &tag_seqs)
{
    cnn::Dict &word_dict = sim->get_input_dict() ;
    cnn::Dict &tag_dict = sim->get_output_dict() ;
    DictWrapper &word_dict_wrapper = sim->get_input_dict_wrapper() ;
    assert(!word_dict.is_frozen() && !tag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, sents, tag_seqs);
    word_dict_wrapper.Freeze();
    word_dict_wrapper.SetUnk(sim->UNK_STR);
    tag_dict.Freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set word , tag dict done . ";
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::read_devel_data(std::istream &is,
                                                       std::vector<IndexSeq> &sents, 
                                                       std::vector<IndexSeq> &tag_seqs)
{
    cnn::Dict &word_dict = sim->get_input_dict() ;
    cnn::Dict &tag_dict = sim->get_output_dict() ;
    assert(word_dict.is_frozen() && tag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, sents, tag_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::read_test_data(std::istream &is,
                                                      std::vector<Seq> &raw_test_sents, 
                                                      std::vector<IndexSeq> &sents)
{
    cnn::Dict &word_dict = sim->get_input_dict() ;
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

template <typename SIModel>
void SingleInputModelHandler<SIModel>::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    sim->set_model_param(varmap) ;
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::build_model()
{
    sim->build_model_structure() ;
    sim->print_model_info() ;
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::train(const std::vector<IndexSeq> *p_sents, 
                                             const std::vector<IndexSeq> *p_tag_seqs,
                                             unsigned max_epoch,
                                             const std::vector<IndexSeq> *p_dev_sents, 
                                             const std::vector<IndexSeq> *p_dev_tag_seqs,
                                             unsigned do_devel_freq,
                                             unsigned trivial_report_freq)
{
    unsigned nr_samples = p_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";
    DictWrapper &word_dict_wrapper = sim->get_input_dict_wrapper() ;
    std::vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    cnn::SimpleSGDTrainer sgd(sim->get_cnn_model());
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq dynamic_sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
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
                sim->build_loss(cg, dynamic_sent_after_replace_unk, tag_seq);
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
        if (p_dev_sents != nullptr)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float F1 = devel(p_dev_sents , p_dev_tag_seqs);
            if (F1 > best_F1) save_current_best_model(F1);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

template <typename SIModel>
float SingleInputModelHandler<SIModel>::devel(const std::vector<IndexSeq> *p_sents, 
                                              const std::vector<IndexSeq> *p_tag_seqs)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    CWSStat stat(sim->get_tag_sys() , true);
    stat.start_time_stat();
    std::vector<IndexSeq> predict_tag_seqs(p_tag_seqs->size());
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        cnn::ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq &sent = p_sents->at(access_idx);
        sim->predict(cg, sent,predict_tag_seq);
        predict_tag_seqs[access_idx] = predict_tag_seq;
        stat.total_tags += predict_tag_seq.size();
    }
    stat.end_time_stat();
    std::array<float , 4> eval_scores = stat.eval(*p_tag_seqs, predict_tag_seqs);
    float Acc = eval_scores[0] , 
        P = eval_scores[1] ,
        R = eval_scores[2] ,
        F1 = eval_scores[3] ;
    std::ostringstream tmp_sos;
    tmp_sos << "validation finished .\n"
        << "Acc = " << Acc << "% , P = " << P << "% , R = " << R << "% , F1 = " << F1 << "%";
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str(tmp_sos.str()) ;
    return F1;
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::predict(std::istream &is, std::ostream &os)
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
        sim->predict(cg, sent, pred_tag_seq);
        Seq words ;
        sim->get_tag_sys().parse_word_tag2words(raw_sent, pred_tag_seq, words) ;
        os << words[0] ;
        for( size_t i = 1 ; i < words.size() ; ++i ) os << OUT_SPLIT_DELIMITER << words[i] ;
        os << "\n";
        stat.total_tags += pred_tag_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    if( best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail() )
    {
        BOOST_LOG_TRIVIAL(info) << "fetch best model ...";
        sim->set_cnn_model(best_model_tmp_ss) ;
    }
    boost::archive::text_oarchive to(os);
    to << *(static_cast<SIModel*>(sim));
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

template <typename SIModel>
void SingleInputModelHandler<SIModel>::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is) ;
    ti >> *(static_cast<SIModel*>(sim));
    sim->print_model_info() ;
}

} // end of namespace slnn
#endif
