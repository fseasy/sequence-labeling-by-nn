#ifndef SLNN_SEGMENTOR_MODEL_HANDLER_INPUT2_MODELHANDLER_H
#define SLNN_SEGMENTOR_MODEL_HANDLER_INPUT2_MODELHANDLER_H

#include <sstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string/split.hpp>
#include "segmentor/base_model/input2_model.h"
#include "segmentor/cws_module/cws_tagging_system.h"

#include "utils/stat.hpp"
namespace slnn{

template <typename I2Model>
class Input2ModelHandler
{
public :
    static const size_t SentMaxLen = 256;
    static const size_t MaxSentNum = 0x8000; // 32k
    static const std::string OUT_SPLIT_DELIMITER ;

public :
    Input2ModelHandler() ;
    ~Input2ModelHandler() ;
    Input2ModelHandler(const Input2ModelHandler &) = delete;
    Input2ModelHandler& operator=(const Input2ModelHandler &) = delete;

    // Before read data
    void set_unk_replace_threshold(int freq_thres , float prob_thres);
    void build_fixed_dict_from_word2vec_file(std::ifstream &is);

    // Reading data 
    void do_read_annotated_dataset(std::istream &is, 
                                   std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
                                   std::vector<IndexSeq> &tag_seqs);
    void read_training_data_and_build_dicts(std::istream &is,
                                            std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
                                            std::vector<IndexSeq> &tag_seqs);
    void read_devel_data(std::istream &is, 
                         std::vector<IndexSeq> &dynamic_sents, std::vector<IndexSeq> &fixed_sents,
                         std::vector<IndexSeq> &tag_seqs);
    void read_test_data(std::istream &is, 
                        std::vector<Seq> &raw_test_sents, 
                        std::vector<IndexSeq> &daynamic_sents, std::vector<IndexSeq> &fixed_sents);

    // After Reading Training data
    void finish_read_training_data(boost::program_options::variables_map &varmap);
    void build_model();
    void load_fixed_embedding(std::istream &is);

    // Train & devel & predict
    void train(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
               const std::vector<IndexSeq> *p_tag_seqs ,
               unsigned max_epoch, 
               const std::vector<IndexSeq> *p_dev_dsents, const std::vector<IndexSeq> *p_dev_fixed_sents,
               const std::vector<IndexSeq> *p_dev_tag_seqs ,
               unsigned do_devel_freq ,
               unsigned trivial_report_freq);
    float devel(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
                const std::vector<IndexSeq> *p_tag_seqs );
    void predict(std::istream &is, std::ostream &os);

    // Save & Load
    void save_model(std::ostream &os);
    void load_model(std::istream &is);

protected :
    inline void save_current_best_model(float F1);

protected :
    Input2Model *i2m ;
    float best_F1;
    std::stringstream best_model_tmp_ss;
};

} // end of namespace slnn

/**********************************************
    implementation for SingleModelHandler
***********************************************/

namespace slnn{

template <typename I2Model>
const size_t Input2ModelHandler<I2Model>::SentMaxLen;

template <typename I2Model>
const size_t Input2ModelHandler<I2Model>::MaxSentNum;

template <typename I2Model>
const std::string Input2ModelHandler<I2Model>::OUT_SPLIT_DELIMITER = "\t" ;

template <typename I2Model>
Input2ModelHandler<I2Model>::Input2ModelHandler()
    : i2m(new I2Model()) ,
    best_F1(0.f) ,
    best_model_tmp_ss()
{}

template <typename I2Model>
Input2ModelHandler<I2Model>::~Input2ModelHandler()
{
    delete i2m ;
}

template<typename I2Model>
inline 
void Input2ModelHandler<I2Model>::save_current_best_model(float F1)
{
    BOOST_LOG_TRIVIAL(info) << "better model has been found . stash it .";
    best_F1 = F1;
    best_model_tmp_ss.str(""); // first , clear it's content !
    boost::archive::text_oarchive to(best_model_tmp_ss);
    to << *i2m->get_cnn_model();
}

template<typename I2Model>
void Input2ModelHandler<I2Model>::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    i2m->get_dynamic_dict_wrapper().set_threshold(freq_thres, prob_thres);
}

template<typename I2Model>
void Input2ModelHandler<I2Model>::build_fixed_dict_from_word2vec_file(std::ifstream &is)
{
    BOOST_LOG_TRIVIAL(info) << "initialize fixed dict .";
    std::string line;
    std::vector<std::string> split_cont;
    getline(is, line); // first line should be the infomation : word-dict-size , word-embedding-dimension
    boost::split(split_cont, line, boost::is_any_of(" "));
    unsigned fixed_dict_sz,
        fixed_word_dim;
    bool is_standard_word2vec_format ;
    if( 2U != split_cont.size() )
    {
        // not standard word2vec file . may be it's the only embedding
        is_standard_word2vec_format = false ;
        fixed_dict_sz = 0 ;
        fixed_word_dim = split_cont.size() - 1 ;
        is.clear();
        is.seekg(0 , is.beg) ;
    }
    else
    {
        is_standard_word2vec_format = true ;
        fixed_dict_sz = std::stol(split_cont[0]) + 1; // another UNK
        fixed_word_dim = std::stol(split_cont[1]);
    }
    cnn::Dict &fixed_dict = i2m->get_fixed_dict() ;
    // read all words and add to dc_m.fixed_dict
    while (getline(is, line))
    {
        std::string::size_type delim_pos = line.find(" ");
        assert(delim_pos != std::string::npos);
        std::string word = line.substr(0, delim_pos);
        fixed_dict.Convert(word);  // add to dict
    }
    //  freeze & add unk to fixed_dict
    fixed_dict.Freeze();
    fixed_dict.SetUnk(i2m->UNK_STR);
    if(!is_standard_word2vec_format)
    {
        // the unstandard word2vec embedding
        fixed_dict_sz = fixed_dict.size() ;
    }
    else
    {
        assert(fixed_dict_sz == fixed_dict.size()) ;
    }
    i2m->set_fixed_word_dict_size_and_embedding(fixed_dict_sz, fixed_word_dim);
    BOOST_LOG_TRIVIAL(info) << "initialize fixed dict done .";
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::load_fixed_embedding(std::istream &is)
{
    // set lookup parameters from outer word embedding
    // using words_loopup_param.Initialize( word_id , value_vector )
    BOOST_LOG_TRIVIAL(info) << "load pre-trained word embedding .";
    std::string line;
    std::vector<std::string> split_cont;
    getline(is, line); // first line is the infomation , skip
    split_cont.reserve(i2m->fixed_word_dim + 1); // word + numbers 
    unsigned long line_cnt = 0; // for warning when read embedding error
    unsigned long words_cnt_hit = 0;
    std::vector<cnn::real> embedding_vec(i2m->fixed_word_dim , 0.f);
    cnn::Dict &fixed_dict = i2m->get_fixed_dict() ;
    cnn::Dict &dynamic_dict = i2m->get_fixed_dict() ;
    cnn::LookupParameters *fixed_lookup_param = i2m->get_fixed_lookup_param() ;
    Index dynamic_unk = dynamic_dict.Convert(i2m->UNK_STR); // for calc hit rate
    while (getline(is, line))
    {
        ++line_cnt;
        boost::trim_right(line);
        boost::split(split_cont, line, boost::is_any_of(" "));
        if (i2m->fixed_word_dim + 1 != split_cont.size())
        {
            BOOST_LOG_TRIVIAL(info) << "bad word dimension : `" << split_cont.size() - 1 << "` at line " << line_cnt;
            continue;
        }
        std::string &word = split_cont.at(0);
        Index word_id = fixed_dict.Convert(word);
        for (size_t idx = 1; idx < split_cont.size(); ++idx)
        {
            embedding_vec[idx - 1] = std::stof(split_cont[idx]);
        }
        fixed_lookup_param->Initialize(word_id, embedding_vec);
        if(dynamic_dict.Convert(word) != dynamic_unk) ++words_cnt_hit;
    }
    BOOST_LOG_TRIVIAL(info) << "load fixed embedding done . hit rate " 
        << words_cnt_hit << "/" << i2m->fixed_dict_size  << " ("
        << ( i2m->fixed_dict_size ? static_cast<float>(words_cnt_hit) / i2m->fixed_dict_size : 0. ) * 100 
        << " %) " ;
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::do_read_annotated_dataset(std::istream &is,
                                                            std::vector<IndexSeq> &dynamic_sents, 
                                                            std::vector<IndexSeq> &fixed_sents,
                                                            std::vector<IndexSeq> &tag_seqs)
{
    unsigned line_cnt = 0;
    std::string line;
    std::vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents,
        tmp_tag_seqs;
    IndexSeq dsent,
        fsent,
        tag_seq;
    // pre-allocation
    tmp_dynamic_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_fixed_sents.reserve(MaxSentNum);
    tmp_tag_seqs.reserve(MaxSentNum);

    dsent.reserve(SentMaxLen);
    fsent.reserve(SentMaxLen);
    tag_seq.reserve(SentMaxLen);

    DictWrapper &dynamic_dict_wrapper = i2m->get_dynamic_dict_wrapper() ;
    cnn::Dict &fixed_dict = i2m->get_fixed_dict();
    cnn::Dict &tag_dict = i2m->get_tag_dict();
    while (getline(is, line)) {
        if (0 == line.size()) continue;
        dsent.clear();
        fsent.clear();
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
                Index dword_id = dynamic_dict_wrapper.Convert(tmp_word_cont[i]) ;
                Index fword_id = fixed_dict.Convert(tmp_word_cont[i]);
                Index tag_id = tag_dict.Convert(tmp_tag_cont[i]) ;
                dsent.push_back(dword_id);
                fsent.push_back(fword_id);
                tag_seq.push_back(tag_id);
            }
        }
        tmp_dynamic_sents.push_back(dsent);
        tmp_fixed_sents.push_back(fsent);
        tmp_tag_seqs.push_back(tag_seq);
        ++line_cnt;
        if(0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    std::swap(dynamic_sents, tmp_dynamic_sents);
    std::swap(fixed_sents, tmp_fixed_sents);
    std::swap(tag_seqs, tmp_tag_seqs);
}
template <typename I2Model>
void Input2ModelHandler<I2Model>::read_training_data_and_build_dicts(std::istream &is,
                                                                     std::vector<IndexSeq> &dsents, 
                                                                     std::vector<IndexSeq> &fsents,
                                                                     std::vector<IndexSeq> &tag_seqs)
{
    cnn::Dict &dword_dict = i2m->get_dynamic_dict() ;
    cnn::Dict &tag_dict = i2m->get_tag_dict() ;
    cnn::Dict &fword_dict = i2m->get_fixed_dict();
    DictWrapper &word_dict_wrapper = i2m->get_dynamic_dict_wrapper() ;

    assert(!dword_dict.is_frozen() && !tag_dict.is_frozen() && fword_dict.is_frozen()); // fixed dict should be frozen already
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, dsents, fsents, tag_seqs);
    word_dict_wrapper.Freeze();
    word_dict_wrapper.SetUnk(i2m->UNK_STR);
    tag_dict.Freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set word , tag dict done . ";
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::read_devel_data(std::istream &is,
                                                  std::vector<IndexSeq> &dsents,
                                                  std::vector<IndexSeq> &fsents,
                                                  std::vector<IndexSeq> &tag_seqs)
{
    cnn::Dict &dword_dict = i2m->get_dynamic_dict() ;
    cnn::Dict &fword_dict = i2m->get_fixed_dict();
    cnn::Dict &tag_dict = i2m->get_tag_dict() ;
    assert(dword_dict.is_frozen() && fword_dict.is_frozen() && tag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, dsents, fsents, tag_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::read_test_data(std::istream &is,
                                                 std::vector<Seq> &raw_test_sents,
                                                 std::vector<IndexSeq> &dsents,
                                                 std::vector<IndexSeq> &fsents)
{
    cnn::Dict &dword_dict = i2m->get_dynamic_dict() ;
    cnn::Dict &fword_dict = i2m->get_fixed_dict() ;
    std::string line ;
    std::vector<Seq> tmp_raw_sents ;
    std::vector<IndexSeq> tmp_dsents,
        tmp_fsents;

    Seq raw_sent ;
    IndexSeq dsent,
        fsent;
    while( getline(is, line) )
    {
        // do not skip empty line .
        CWSTaggingSystem::split_word(line, raw_sent) ;
        dsent.clear();
        fsent.clear();
        for( size_t i = 0 ; i < raw_sent.size() ; ++i )
        {
            dsent.push_back(dword_dict.Convert(raw_sent[i]));
            fsent.push_back(fword_dict.Convert(raw_sent[i]));
        }
        tmp_raw_sents.push_back(raw_sent) ;
        tmp_dsents.push_back(dsent);
        tmp_fsents.push_back(fsent);
    }
    std::swap(raw_test_sents, tmp_raw_sents) ;
    std::swap(dsents, tmp_dsents);
    std::swap(fsents, tmp_fsents);
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    i2m->set_model_param(varmap) ;
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::build_model()
{
    i2m->build_model_structure() ;
    i2m->print_model_info() ;
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::train(const std::vector<IndexSeq> *p_dsents,
                                        const std::vector<IndexSeq> *p_fsents,
                                        const std::vector<IndexSeq> *p_tag_seqs,
                                        unsigned max_epoch,
                                        const std::vector<IndexSeq> *p_dev_dsents,
                                        const std::vector<IndexSeq> *p_dev_fsents,
                                        const std::vector<IndexSeq> *p_dev_tag_seqs,
                                        unsigned do_devel_freq,
                                        unsigned trivial_report_freq)
{
    unsigned nr_samples = p_dsents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";
    DictWrapper &dynamic_dict_wrapper = i2m->get_dynamic_dict_wrapper() ;
    cnn::Dict &fdict = i2m->get_fixed_dict() ;

    std::vector<unsigned> access_order(nr_samples);
    for( unsigned i = 0; i < nr_samples; ++i ) access_order[i] = i;

    cnn::SimpleSGDTrainer sgd(i2m->get_cnn_model());
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq dynamic_sent_after_replace_unk(SentMaxLen, 0);
    for( unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch )
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
            const IndexSeq &dsent = p_dsents->at(access_idx),
                &fsent = p_fsents->at(access_idx),
                &tag_seq = p_tag_seqs->at(access_idx);
            { // new scope , for only one Computatoin Graph can be exists in one scope at the same time .
              // devel will creat another Computation Graph , so we need to create new scoce to release it before devel .
                cnn::ComputationGraph cg ;
                dynamic_sent_after_replace_unk.resize(dsent.size());
                for( size_t word_idx = 0; word_idx < dsent.size(); ++word_idx )
                {
                    dynamic_sent_after_replace_unk[word_idx] =
                        dynamic_dict_wrapper.ConvertProbability(dsent.at(word_idx));
                }
                i2m->build_loss(cg, dynamic_sent_after_replace_unk, fsent, tag_seq);
                cnn::real loss = as_scalar(cg.forward());
                cg.backward();
                sgd.update(1.f);
                training_stat_per_epoch.loss += loss;
                training_stat_per_epoch.total_tags += dsent.size() ;
            }
            if( 0 == (i + 1) % trivial_report_freq ) // Report 
            {
                std::string trivial_header = std::to_string(i + 1) + " instances have been trained.";
                BOOST_LOG_TRIVIAL(trace) << training_stat_per_epoch.get_stat_str(trivial_header);
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if( p_dev_dsents != nullptr && 0 == line_cnt_for_devel % do_devel_freq )
            {
                float F1 = devel(p_dev_dsents, p_dev_fsents, p_dev_tag_seqs);
                if( F1 > best_F1 ) save_current_best_model(F1);
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
        if( p_dev_dsents != nullptr )
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float F1 = devel(p_dev_dsents, p_dev_fsents, p_dev_tag_seqs);
            if( F1 > best_F1 ) save_current_best_model(F1);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

template <typename I2Model>
float Input2ModelHandler<I2Model>::devel(const std::vector<IndexSeq> *p_dsents, 
                                         const std::vector<IndexSeq> *p_fsents,
                                              const std::vector<IndexSeq> *p_tag_seqs)
{
    unsigned nr_samples = p_dsents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    CWSStat stat(i2m->get_tag_sys() , true);
    stat.start_time_stat();
    std::vector<IndexSeq> predict_tag_seqs(p_tag_seqs->size());
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        cnn::ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq &dsent = p_dsents->at(access_idx),
            &fsent = p_fsents->at(access_idx);
        i2m->predict(cg, dsent, fsent, predict_tag_seq);
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

template <typename I2Model>
void Input2ModelHandler<I2Model>::predict(std::istream &is, std::ostream &os)
{
    
    std::vector<Seq> raw_instances;
    std::vector<IndexSeq> dsents,
        fsents;
    read_test_data(is,raw_instances, dsents, fsents );
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
        IndexSeq &dsent = dsents.at(i),
            &fsent = fsents.at(i);
        IndexSeq pred_tag_seq;
        cnn::ComputationGraph cg;
        i2m->predict(cg, dsent, fsent, pred_tag_seq);
        Seq words ;
        i2m->get_tag_sys().parse_word_tag2words(raw_sent, pred_tag_seq, words) ;
        os << words[0] ;
        for( size_t i = 1 ; i < words.size() ; ++i ) os << OUT_SPLIT_DELIMITER << words[i] ;
        os << "\n";
        stat.total_tags += pred_tag_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    if( best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail() )
    {
        BOOST_LOG_TRIVIAL(info) << "fetch best model ...";
        i2m->set_cnn_model(best_model_tmp_ss) ;
    }
    boost::archive::text_oarchive to(os);
    to << *i2m ;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

template <typename I2Model>
void Input2ModelHandler<I2Model>::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is) ;
    ti >> *i2m ;
    i2m->print_model_info() ;
}

} // end of namespace slnn
#endif
