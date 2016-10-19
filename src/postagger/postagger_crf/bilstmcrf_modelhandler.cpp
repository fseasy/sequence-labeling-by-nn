
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "utils/typedeclaration.h"
#include "bilstmcrf_modelhandler.h"

using namespace std;
using namespace dynet;
namespace slnn
{


const std::string BILSTMCRFModelHandler::number_transform_str = "##";
const size_t BILSTMCRFModelHandler::length_transform_str = number_transform_str.length();

BILSTMCRFModelHandler::BILSTMCRFModelHandler(BILSTMCRFModel4POSTAG &dc_m) 
    :dc_m(dc_m) , best_acc(0.f) , best_model_tmp_ss()
{}


void BILSTMCRFModelHandler::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    dc_m.word_dict_wrapper.set_threshold(freq_thres, prob_thres);
}

void BILSTMCRFModelHandler::do_read_annotated_dataset(istream &is, vector<IndexSeq> &sents,
    vector<IndexSeq> &postag_seqs)
{
    unsigned line_cnt = 0;
    string line;
    vector<IndexSeq> tmp_sents,
        tmp_postag_seqs;
    IndexSeq sent, 
        postag_seq;
    // pre-allocation
    tmp_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_postag_seqs.reserve(MaxSentNum);
    sent.reserve(SentMaxLen);
    postag_seq.reserve(SentMaxLen);

    while (getline(is, line)) {
        boost::algorithm::trim(line);
        if (0 == line.size()) continue;
        vector<string> strpair_cont;
        boost::algorithm::split(strpair_cont, line, boost::is_any_of("\t"));
        sent.clear();
        postag_seq.clear();
        for (string &strpair : strpair_cont) {
            string::size_type  delim_pos = strpair.rfind("_");
            assert(delim_pos != string::npos);
            std::string word = strpair.substr(0, delim_pos);
            // Parse Number to specific string
            word = replace_number(word);
            Index word_id = dc_m.word_dict_wrapper.convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
            Index postag_id = dc_m.postag_dict.convert(strpair.substr(delim_pos + 1));
            sent.push_back(word_id);
            postag_seq.push_back(postag_id);
        }
        tmp_sents.push_back(sent);
        tmp_postag_seqs.push_back(postag_seq);
        ++line_cnt;
        if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    swap(sents, tmp_sents);
    swap(postag_seqs, tmp_postag_seqs);
}

void BILSTMCRFModelHandler::read_training_data_and_build_word_and_postag_dicts(istream &is, vector<IndexSeq> &sents, 
    vector<IndexSeq> &postag_seqs)
{
    assert(!dc_m.word_dict.is_frozen() && !dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, sents, postag_seqs);
    dc_m.word_dict_wrapper.freeze();
    dc_m.word_dict_wrapper.set_unk(dc_m.UNK_STR);
    dc_m.postag_dict.freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set dynamic dict done . ";
}

void BILSTMCRFModelHandler::read_devel_data(istream &is, vector<IndexSeq> &sents, 
    vector<IndexSeq> &postag_seqs)
{
    assert(dc_m.word_dict.is_frozen() && dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, sents,  postag_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

void BILSTMCRFModelHandler::read_test_data(istream &is, vector<Seq> &raw_test_sents, vector<IndexSeq> &sents)
{
    assert(dc_m.word_dict.is_frozen() && dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read test data .";
    vector<Seq> tmp_raw_sents;
    vector<IndexSeq> tmp_sents;
    string line;
    while (getline(is, line))
    {
        boost::trim(line);
        vector<string> words_seq;
        boost::split(words_seq, line, boost::is_any_of("\t"));
        tmp_raw_sents.push_back(words_seq);
        unsigned seq_len = words_seq.size();
        tmp_sents.emplace_back(seq_len); // using constructor `vector(nr_num)` => push_back(vector<int>(nr_words)) 
        IndexSeq &words_index_seq = tmp_sents.back();
        for (unsigned i = 0; i < seq_len; ++i)
        {
            string number_transed_word = replace_number(words_seq[i]);
            words_index_seq[i] = dc_m.word_dict.convert(number_transed_word);
        }

    }
    swap(tmp_raw_sents, raw_test_sents);
    swap(tmp_sents, sents);
}

void BILSTMCRFModelHandler::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    // set param 
    dc_m.word_embedding_dim = varmap["word_embedding_dim"].as<unsigned>();
    dc_m.postag_embedding_dim = varmap["postag_embedding_dim"].as<unsigned>();
    dc_m.nr_lstm_stacked_layer = varmap["nr_lstm_stacked_layer"].as<unsigned>();
    dc_m.lstm_h_dim = varmap["lstm_h_dim"].as<unsigned>();
    dc_m.merge_hidden_dim = varmap["merge_hidden_dim"].as<unsigned>();

    dc_m.word_dict_size = dc_m.word_dict.size();
    dc_m.postag_dict_size = dc_m.postag_dict.size();
}

void BILSTMCRFModelHandler::build_model()
{
    dc_m.build_model_structure();
    dc_m.print_model_info();
}


void BILSTMCRFModelHandler::train(const vector<IndexSeq> *p_sents, 
    const vector<IndexSeq> *p_postag_seqs,
    unsigned max_epoch,
    const vector<IndexSeq> *p_dev_sents, 
    const vector<IndexSeq> *p_dev_postag_seqs,
    unsigned do_devel_freq , 
    bool do_train_stat ,
    unsigned verbose_train_report )
{
    unsigned nr_samples = p_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";

    vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    SimpleSGDTrainer sgd = SimpleSGDTrainer(dc_m.m);
    unsigned long line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *rndeng);

        // For loss , accuracy , time cost report
        Stat training_stat_per_epoch ;
        shared_ptr<Stat> p_training_stat_per_report(nullptr) ;
        if( do_train_stat ) p_training_stat_per_report.reset(new Stat()) ;

        // training for an epoch
        if(p_training_stat_per_report) p_training_stat_per_report->start_time_stat();
        training_stat_per_epoch.start_time_stat();
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq *p_sent = &p_sents->at(access_idx),
                *p_tag_seq = &p_postag_seqs->at(access_idx);
            ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                           // so we have to declaring it as pointer and destroy it handly 
                                                           // before develing.
                                                           // transform low-frequent words to UNK according to the probability
            sent_after_replace_unk.resize(p_sent->size());
            for (size_t word_idx = 0; word_idx < p_sent->size(); ++word_idx)
            {
                sent_after_replace_unk[word_idx] = dc_m.word_dict_wrapper.unk_replace_probability(p_sent->at(word_idx));
            }
            auto loss_expr = dc_m.viterbi_train(cg, &sent_after_replace_unk, p_tag_seq,
                                p_training_stat_per_report.get());
            dynet::real E = as_scalar(cg->forward(loss_expr));
            cg->backward(loss_expr);
            sgd.update(1.0);
            delete cg;
            
            if(do_train_stat) p_training_stat_per_report->loss += E ;
            else training_stat_per_epoch.loss += E ;
            if (do_train_stat && 0 == (i + 1) % verbose_train_report) // Report 
            {
                p_training_stat_per_report->end_time_stat();
                BOOST_LOG_TRIVIAL(trace) << i + 1 << " instances have been trained , with E = "
                    << p_training_stat_per_report->get_E()
                    << " , ACC = " << p_training_stat_per_report->get_acc() * 100
                    << " % with time cost " << p_training_stat_per_report->get_time_cost_in_seconds()
                    << " s .";
                training_stat_per_epoch += *p_training_stat_per_report;
                p_training_stat_per_report->clear();
                p_training_stat_per_report->start_time_stat();
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if (p_dev_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
            {
                float acc = devel(p_dev_sents, p_dev_postag_seqs);
                if (acc > best_acc) save_current_best_model(acc);
                line_cnt_for_devel = 0; // avoid overflow
            }
        }

        // End of an epoch 
        sgd.update_epoch();

        training_stat_per_epoch.end_time_stat();
        if(do_train_stat) training_stat_per_epoch += *p_training_stat_per_report;

        // Output
        long long epoch_time_cost = training_stat_per_epoch.get_time_cost_in_seconds();
        BOOST_LOG_TRIVIAL(info) << "-------- epoch " << nr_epoch + 1 << "/" << max_epoch << " finished . ----------\n"
            << nr_samples << " instances has been trained ." ;
        if(do_train_stat)
        {
            BOOST_LOG_TRIVIAL(info)   << "For this epoch , E = " << training_stat_per_epoch.get_E() << "\n"
                                      << "ACC = " << training_stat_per_epoch.get_acc() * 100 << "  % \n"
                                      << "total time cost " << epoch_time_cost << " s ( speed "
                                      << training_stat_per_epoch.get_speed_as_kilo_tokens_per_sencond() << " k/s tokens).\n"
                                      << "total tags : " << training_stat_per_epoch.total_tags << " with correct tags : "
                                        << training_stat_per_epoch.correct_tags << " \n";
        }
        else
        {
            BOOST_LOG_TRIVIAL(info) << "For this epoch , \n"
                                    << "Total E = " << training_stat_per_epoch.get_sum_E() << "\n" 
                                    << "Time cost : " << epoch_time_cost << " s" ;
        }
        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();

        // do validation at every ends of epoch
        if (p_dev_sents != nullptr)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float acc = devel(p_dev_sents , p_dev_postag_seqs);
            if (acc > best_acc) save_current_best_model(acc);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with cost " << total_time_cost_in_seconds << " s .";
}

float BILSTMCRFModelHandler::devel(const std::vector<IndexSeq> *p_sents,
    const std::vector<IndexSeq> *p_postag_seqs,
    std::ostream *p_error_output_os)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .\n";
    unsigned long line_cnt4error_output = 0;
    if (p_error_output_os) *p_error_output_os << "line_nr\tword_index\tword_at_dict\tpredict_tag\ttrue_tag\n";
    Stat acc_stat;
    acc_stat.start_time_stat();
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        ++line_cnt4error_output;
        ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq *p_sent = &p_sents->at(access_idx),
            *p_tag_seq = &p_postag_seqs->at(access_idx);
        dc_m.viterbi_predict(&cg, p_sent, &predict_tag_seq);
        assert(predict_tag_seq.size() == p_tag_seq->size());
        for (unsigned i = 0; i < p_tag_seq->size(); ++i)
        {
            ++acc_stat.total_tags;
            if (p_tag_seq->at(i) == predict_tag_seq[i]) ++acc_stat.correct_tags;
            else if (p_error_output_os)
            {
                *p_error_output_os << line_cnt4error_output << "\t" << i << "\t" << dc_m.word_dict.convert(p_sent->at(i))
                    << "\t" << dc_m.postag_dict.convert(predict_tag_seq[i]) << "\t" << dc_m.postag_dict.convert(p_tag_seq->at(i)) << "\n";
            }
        }
    }
    acc_stat.end_time_stat();
    BOOST_LOG_TRIVIAL(info) << "validation finished . ACC = "
        << acc_stat.get_acc() * 100 << " % "
        << ", with time cosing " << acc_stat.get_time_cost_in_seconds() << " s . "
        << "(speed " << acc_stat.get_speed_as_kilo_tokens_per_sencond() << " k/s tokens) "
        << "total tags : " << acc_stat.total_tags << " correct tags : " << acc_stat.correct_tags;
    return acc_stat.get_acc();
}

void BILSTMCRFModelHandler::predict(std::istream &is, std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "do predict ";
    const string SPLIT_DELIMITER = "\t";
    vector<Seq> raw_instances;
    vector<IndexSeq> sents;
    read_test_data(is,raw_instances,sents );
    BOOST_LOG_TRIVIAL(info) << "read " << raw_instances.size() << " instance .";
    Stat time_stat;
    time_stat.start_time_stat();
    for (unsigned int i = 0; i < raw_instances.size(); ++i)
    {
        vector<string> *p_raw_sent = &raw_instances.at(i);
        if (0 == p_raw_sent->size())
        {
            os << "\n";
            continue;
        }
        IndexSeq *p_sent = &sents.at(i);
        IndexSeq predict_seq;
        ComputationGraph cg;
        dc_m.viterbi_predict(&cg, p_sent, &predict_seq);
        // output the result directly
        os << p_raw_sent->at(0) << "_" << dc_m.postag_dict.convert(predict_seq.at(0));
        for (unsigned k = 1; k < p_raw_sent->size(); ++k)
        {
            os << SPLIT_DELIMITER
                << p_raw_sent->at(k) << "_" << dc_m.postag_dict.convert(predict_seq.at(k));
        }
        os << "\n";
    }
    time_stat.end_time_stat();
    BOOST_LOG_TRIVIAL(info) << "predicted done with time costing " << time_stat.get_time_cost_in_seconds() << " s .";
}

void BILSTMCRFModelHandler::save_model(std::ostream &os)
{
    boost::archive::text_oarchive to(os);
    to << dc_m.word_embedding_dim << dc_m.postag_embedding_dim
        << dc_m.nr_lstm_stacked_layer << dc_m.lstm_h_dim
        << dc_m.merge_hidden_dim 
        << dc_m.word_dict_size
        << dc_m.postag_dict_size;

    to << dc_m.word_dict << dc_m.postag_dict;
    if (best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail())
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *dc_m.m;
        ; // if best model is not save to the temporary stringstream , we should firstly save it !
    }

    to << *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

void BILSTMCRFModelHandler::load_model(std::istream &is)
{
    boost::archive::text_iarchive ti(is);
    ti >> dc_m.word_embedding_dim >> dc_m.postag_embedding_dim
        >> dc_m.nr_lstm_stacked_layer >> dc_m.lstm_h_dim
        >> dc_m.merge_hidden_dim 
        >> dc_m.word_dict_size
        >> dc_m.postag_dict_size;

    ti >> dc_m.word_dict >> dc_m.postag_dict;
    assert(dc_m.word_dict_size == dc_m.word_dict.size()
        && dc_m.postag_dict_size == dc_m.postag_dict.size());
    dc_m.build_model_structure();
    ti >> *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "load model done .";
}

} // end of namespace
