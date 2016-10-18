#ifndef SLNN_NER_MODELHANDLER_INPUT2D_MODELHANDLER_H
#define SLNN_NER_MODELHANDLER_INPUT2D_MODELHANDLER_H

#include <boost/algorithm/string/split.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "ner/base_model/input2D_model.h"

#include "utils/stat.hpp"
namespace slnn{

template <typename SIModel>
class Input2DModelHandler
{
public :
    Input2DModel *sim ;

    float best_F1;
    std::stringstream best_model_tmp_ss;

    const size_t SentMaxLen = 256;
    const size_t MaxSentNum = 0x8000; // 32k
    static const std::string OUT_SPLIT_DELIMITER ;
    static const std::string number_transform_str ;
    static const size_t length_transform_str ;

    Input2DModelHandler() ;
    ~Input2DModelHandler() ;
    // Before read data
    void set_unk_replace_threshold(int freq_thres , float prob_thres);
    
    // Reading data 
    std::string replace_number(const std::string &str) ;
    void do_read_annotated_dataset(std::istream &is, std::vector<IndexSeq> &sents,
                                   std::vector<IndexSeq> &postag_seqs , std::vector<IndexSeq> &ner_seqs);
    void read_training_data_and_build_dicts(std::istream &is, std::vector<IndexSeq> &sents, 
                                            std::vector<IndexSeq> &postag_seqs , std::vector<IndexSeq> &ner_seqs);
    void read_devel_data(std::istream &is, std::vector<IndexSeq> &sents, 
                         std::vector<IndexSeq> &postag_seqs , std::vector<IndexSeq> &ner_seqs);
    void read_test_data(std::istream &is, std::vector<Seq> &raw_test_sents, std::vector<IndexSeq> &sents ,
                        std::vector<IndexSeq> &postag_seqs);

    // After Reading Training data
    void finish_read_training_data(boost::program_options::variables_map &varmap);
    void build_model();

    // Train & devel & predict
    void train(const std::vector<IndexSeq> *p_sents, 
               const std::vector<IndexSeq> *p_postag_seqs , const std::vector<IndexSeq> *p_ner_seqs ,
               unsigned max_epoch, 
               const std::vector<IndexSeq> *p_dev_sents, 
               const std::vector<IndexSeq> *p_dev_postag_seqs , const std::vector<IndexSeq> *p_dev_ner_seqs ,
               const std::string *p_conlleval_script_path,
               unsigned do_devel_freq ,
               unsigned trivial_report_freq);
    float devel(const std::vector<IndexSeq> *p_sents, const std::vector<IndexSeq> *p_postag_seqs ,
                const std::vector<IndexSeq> *p_ner_seqs ,
                const std::string *p_conlleval_script_path);
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);
protected :
    void save_current_best_model(float F1);
    bool is_train_error_occurs(float cur_F1);
};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{
template <typename SIModel>
const std::string Input2DModelHandler<SIModel>::OUT_SPLIT_DELIMITER = "\t" ;

template <typename SIModel>
const std::string Input2DModelHandler<SIModel>::number_transform_str = "##";

template <typename SIModel>
const size_t Input2DModelHandler<SIModel>::length_transform_str = number_transform_str.length();

template <typename SIModel>
Input2DModelHandler<SIModel>::Input2DModelHandler()
    : sim(new SIModel()) ,
    best_F1(0.f) ,
    best_model_tmp_ss()
{}

template <typename SIModel>
Input2DModelHandler<SIModel>::~Input2DModelHandler()
{
    delete sim ;
}

template<typename SIModel>
inline 
void Input2DModelHandler<SIModel>::save_current_best_model(float F1)
{
    BOOST_LOG_TRIVIAL(info) << "better model has been found . stash it .";
    best_F1 = F1 ;
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss);
    to << *sim->get_dynet_model();
}

template<typename SIModel>
inline 
bool Input2DModelHandler<SIModel>::is_train_error_occurs(float cur_F1)
{
    return  (best_F1 - cur_F1 > 20.f);
}

template<typename SIModel>
void Input2DModelHandler<SIModel>::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    sim->get_word_dict_wrapper().set_threshold(freq_thres, prob_thres);
}

template<typename SIModel>
std::string Input2DModelHandler<SIModel>::replace_number(const std::string &str)
{
    std::string tmp_str = str;
    size_t start_pos = 0;
    while (start_pos < tmp_str.length())
    {
        size_t end_pos = start_pos;

        while (true)
        {
            size_t byte_len = UTF8Processing::get_number_byte_width(tmp_str, end_pos);
            if (0 == byte_len) break;
            else end_pos += byte_len;
        }
        size_t number_byte_len = end_pos - start_pos;
        if (0 != number_byte_len)
        {
            // replace
            tmp_str.replace(start_pos, number_byte_len, number_transform_str);
            start_pos += length_transform_str;
        }
        else ++start_pos;
    }
    return tmp_str;
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::do_read_annotated_dataset(std::istream &is,
                                                             std::vector<IndexSeq> &sents, 
                                                             std::vector<IndexSeq> &postag_seqs ,
                                                             std::vector<IndexSeq> &ner_seqs)
{
    DictWrapper &word_dict_wrapper = sim->get_word_dict_wrapper() ;
    dynet::Dict &postag_dict = sim->get_postag_dict() ;
    dynet::Dict &ner_dict = sim->get_ner_dict() ;
    unsigned line_cnt = 0;
    std::string line;
    std::vector<IndexSeq> tmp_sents,
            tmp_postag_seqs,
            tmp_ner_seqs;
    IndexSeq sent, 
        postag_seq,
        ner_seq;
    // pre-allocation
    tmp_sents.reserve(MaxSentNum); // 32k pairs 
    tmp_postag_seqs.reserve(MaxSentNum);
    tmp_ner_seqs.reserve(MaxSentNum);

    sent.reserve(SentMaxLen);
    postag_seq.reserve(SentMaxLen);
    ner_seq.reserve(SentMaxLen);

    while (getline(is, line)) {
        boost::algorithm::trim(line);
        if (0 == line.size()) continue;
        std::vector<std::string> parts;
        boost::algorithm::split(parts, line, boost::is_any_of("\t"));
        sent.clear();
        postag_seq.clear();
        ner_seq.clear();
        for (std::string &part : parts) {
            std::string::size_type postag_pos = part.rfind("/");
            std::string::size_type nertag_pos = part.rfind("#");
            assert(postag_pos != std::string::npos && nertag_pos != std::string::npos);
            std::string word = part.substr(0, postag_pos);
            std::string postag = part.substr(postag_pos + 1, nertag_pos - postag_pos - 1);
            std::string nertag = part.substr(nertag_pos + 1);
            // Parse Number to specific string
            word = replace_number(word);
            Index word_id = word_dict_wrapper.Convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
            Index postag_id = postag_dict.Convert(postag);
            Index nertag_id = ner_dict.Convert(nertag);
            sent.push_back(word_id);
            postag_seq.push_back(postag_id);
            ner_seq.push_back(nertag_id);
        }
        tmp_sents.push_back(sent);
        tmp_postag_seqs.push_back(postag_seq);
        tmp_ner_seqs.push_back(ner_seq);
        ++line_cnt;
        if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    swap(sents, tmp_sents);
    swap(postag_seqs, tmp_postag_seqs);
    swap(ner_seqs, tmp_ner_seqs);
}


template <typename SIModel>
void Input2DModelHandler<SIModel>::read_training_data_and_build_dicts(std::istream &is,
                                                                      std::vector<IndexSeq> &sents, 
                                                                      std::vector<IndexSeq> &postag_seqs ,
                                                                      std::vector<IndexSeq> &ner_seqs)
{
    DictWrapper &word_dict_wrapper = sim->get_word_dict_wrapper() ;
    dynet::Dict &postag_dict = sim->get_postag_dict() ;
    dynet::Dict &ner_dict = sim->get_ner_dict() ;
    assert(!word_dict_wrapper.is_frozen() && !postag_dict.is_frozen() && !ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, sents, postag_seqs , ner_seqs);
    word_dict_wrapper.Freeze();
    word_dict_wrapper.SetUnk(sim->UNK_STR);
    postag_dict.Freeze();
    ner_dict.Freeze() ;
    BOOST_LOG_TRIVIAL(info) << "read training data done and set word , tag dict done . ";
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::read_devel_data(std::istream &is,
                                                   std::vector<IndexSeq> &sents,
                                                   std::vector<IndexSeq> &postag_seqs,
                                                   std::vector<IndexSeq> &ner_seqs)
{
    dynet::Dict &word_dict = sim->get_word_dict() ;
    dynet::Dict &postag_dict = sim->get_postag_dict() ;
    dynet::Dict &ner_dict = sim->get_ner_dict() ;
    assert(word_dict.is_frozen() && postag_dict.is_frozen() && ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, sents, postag_seqs , ner_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::read_test_data(std::istream &is,
                                                  std::vector<Seq> &raw_test_sents, 
                                                  std::vector<IndexSeq> &sents ,
                                                  std::vector<IndexSeq> &postag_seqs)
{
    dynet::Dict &word_dict = sim->get_word_dict() ;
    dynet::Dict &postag_dict = sim->get_postag_dict() ;
    assert(word_dict.is_frozen() && postag_dict.is_frozen()) ;
    BOOST_LOG_TRIVIAL(info) << "read test data .";
    std::vector<Seq> tmp_raw_sents;
    std::vector<IndexSeq> tmp_sents,
        tmp_postag_seqs;
    std::string line;
    while (getline(is, line))
    {
        boost::trim(line);
        std::vector<std::string> parts_seq;
        boost::split(parts_seq, line, boost::is_any_of("\t"));
        unsigned seq_len = parts_seq.size();
        tmp_raw_sents.emplace_back(seq_len);
        tmp_sents.emplace_back(seq_len); // using constructor `vector(nr_num)` => push_back(vector<int>(nr_words)) 
        tmp_postag_seqs.emplace_back(seq_len);
        Seq &raw_sent = tmp_raw_sents.back();
        IndexSeq &words_index_seq = tmp_sents.back() ,
            &postag_index_seq = tmp_postag_seqs.back();
        for (unsigned i = 0; i < seq_len; ++i)
        {
            std::string &part = parts_seq.at(i);
            std::string::size_type delim_pos = part.rfind("_");
            std::string raw_word = part.substr(0, delim_pos);
            std::string postag = part.substr(delim_pos + 1);
            std::string number_transed_word = replace_number(raw_word);
            raw_sent[i] = raw_word ;
            words_index_seq[i] = word_dict.Convert(number_transed_word);
            postag_index_seq[i] = postag_dict.Convert(postag);
        }

    }
    swap(tmp_raw_sents, raw_test_sents);
    swap(tmp_sents, sents);
    swap(tmp_postag_seqs, postag_seqs);
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    sim->set_model_param(varmap) ;
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::build_model()
{
    sim->build_model_structure() ;
    sim->print_model_info() ;
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::train(const std::vector<IndexSeq> *p_sents, 
                                         const std::vector<IndexSeq> *p_postag_seqs,
                                         const std::vector<IndexSeq> *p_ner_seqs ,
                                         unsigned max_epoch,
                                         const std::vector<IndexSeq> *p_dev_sents, 
                                         const std::vector<IndexSeq> *p_dev_postag_seqs,
                                         const std::vector<IndexSeq> *p_dev_ner_seqs,
                                         const std::string *p_conlleval_script_path,
                                         unsigned do_devel_freq,
                                         unsigned trivial_report_freq)
{
    unsigned nr_samples = p_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";
    DictWrapper &word_dict_wrapper = sim->get_word_dict_wrapper() ;
    std::vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    bool is_train_ok = true;
    dynet::SimpleSGDTrainer sgd(sim->get_dynet_model());
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch && is_train_ok; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *dynet::rndeng);

        // For loss , accuracy , time cost report
        BasicStat training_stat_per_epoch;
        training_stat_per_epoch.start_time_stat();

        // train for every Epoch 
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq &sent = p_sents->at(access_idx),
                &postag_seq = p_postag_seqs->at(access_idx) ,
                &ner_seq = p_ner_seqs->at(access_idx);
            { // new scope , for only one Computatoin Graph can be exists in one scope at the same time .
              // devel will creat another Computation Graph , so we need to create new scoce to release it before devel .
                dynet::ComputationGraph cg ;
                sent_after_replace_unk.resize(sent.size());
                for( size_t word_idx = 0; word_idx < sent.size(); ++word_idx )
                {
                    sent_after_replace_unk[word_idx] =
                        word_dict_wrapper.ConvertProbability(sent.at(word_idx));
                }
                sim->build_loss(cg, sent_after_replace_unk, postag_seq , ner_seq );
                dynet::real loss = as_scalar(cg.forward());
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
                float F1 = devel(p_dev_sents  , p_dev_postag_seqs , p_dev_ner_seqs, p_conlleval_script_path);
                if (F1 > best_F1) save_current_best_model(F1);
                line_cnt_for_devel = 0; // avoid overflow
                if( is_train_error_occurs(F1) )
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
        if (p_dev_sents != nullptr && is_train_ok)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float F1 = devel(p_dev_sents , p_dev_postag_seqs , p_dev_ner_seqs, p_conlleval_script_path);
            if (F1 > best_F1) save_current_best_model(F1);
            if( is_train_error_occurs(F1) )
            {
                is_train_ok = false;
                break;
            }
        }
    }
    if( !is_train_ok ){ BOOST_LOG_TRIVIAL(warning) << "Gradient may have been updated error ! Exit ahead of time." ; }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

template <typename SIModel>
float Input2DModelHandler<SIModel>::devel(const std::vector<IndexSeq> *p_sents, 
                                          const std::vector<IndexSeq> *p_postag_seqs,
                                          const std::vector<IndexSeq> *p_ner_seqs,
                                          const std::string *p_conlleval_script_path)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    NerStat stat(*p_conlleval_script_path, "eval_out.tmp", true) ;
    std::vector<IndexSeq> pred_ner_seqs(p_ner_seqs->size());
    stat.start_time_stat();
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        dynet::ComputationGraph cg;
        IndexSeq predict_ner_seq;
        const IndexSeq &sent = p_sents->at(access_idx) ,
            &postag_seq = p_postag_seqs->at(access_idx);
        sim->predict(cg, sent, postag_seq, predict_ner_seq);
        stat.total_tags += predict_ner_seq.size();
        pred_ner_seqs[access_idx] = predict_ner_seq;
    }
    stat.end_time_stat();
  
    std::array<float , 4> eval_scores = stat.conlleval(*p_ner_seqs, pred_ner_seqs, sim->get_ner_dict());
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
void Input2DModelHandler<SIModel>::predict(std::istream &is, std::ostream &os)
{
    
    std::vector<Seq> raw_instances;
    std::vector<IndexSeq> sents ,
        postag_seqs;
    read_test_data(is,raw_instances, sents , postag_seqs);
    BOOST_LOG_TRIVIAL(info) << "do prediction on " << raw_instances.size() << " instances .";
    BasicStat stat(true);
    dynet::Dict &postag_dict = sim->get_postag_dict() ;
    dynet::Dict &ner_dict = sim->get_ner_dict() ;
    stat.start_time_stat();
    for (unsigned int i = 0; i < raw_instances.size(); ++i)
    {
        Seq &raw_sent = raw_instances.at(i);
        if (0 == raw_sent.size())
        {
            os << "\n";
            continue;
        }
        IndexSeq &sent = sents.at(i) ,
            postag_seq = postag_seqs.at(i);
        IndexSeq pred_ner_seq;
        dynet::ComputationGraph cg;
        sim->predict(cg, sent, postag_seq, pred_ner_seq);
        os << raw_sent[0] 
            << "/" << postag_dict.Convert(postag_seq[0]) 
            << "#" << ner_dict.Convert(pred_ner_seq[0]);
        for( size_t i = 1 ; i < raw_sent.size() ; ++i )
        { 
            os << OUT_SPLIT_DELIMITER 
                << raw_sent[i] 
                << "/" << postag_dict.Convert(postag_seq[i]) 
                << "#" << ner_dict.Convert(pred_ner_seq[i]); 
        }
        os << "\n";
        stat.total_tags += pred_ner_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::save_model(std::ostream &os)
{
    if( best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail() )
    {
        sim->set_dynet_model(best_model_tmp_ss) ;
    }
    sim->save_model(os) ;
}

template <typename SIModel>
void Input2DModelHandler<SIModel>::load_model(std::istream &is)
{
    sim->load_model(is) ;
    sim->print_model_info() ;
}

} // end of namespace slnn
#endif
