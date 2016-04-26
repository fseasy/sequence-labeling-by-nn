
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "utils/typedeclaration.h"
#include "doublechannel_modelhandler.h"

using namespace std;
using namespace cnn;
namespace slnn
{

DoubleChannelModelHandler::DoubleChannelModelHandler(DoubleChannelModel4POSTAG &dc_m) 
    :dc_m(dc_m)
{}

void DoubleChannelModelHandler::do_read_annotated_dataset(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
    vector<IndexSeq> &postag_seqs)
{
    unsigned line_cnt = 0;
    string line;
    vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents,
        tmp_postag_seqs;
    IndexSeq dynamic_sent, 
        fixed_sent ,
        postag_seq;
    // pre-allocation
    tmp_dynamic_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_fixed_sents.reserve(MaxSentNum);
    tmp_postag_seqs.reserve(MaxSentNum);
    dynamic_sent.reserve(SentMaxLen);
    fixed_sent.reserve(SentMaxLen);
    postag_seq.reserve(SentMaxLen);

    while (getline(is, line)) {
        boost::algorithm::trim(line);
        if (0 == line.size()) continue;
        vector<string> strpair_cont;
        boost::algorithm::split(strpair_cont, line, boost::is_any_of("\t"));
        dynamic_sent.clear();
        fixed_sent.clear();
        postag_seq.clear();
        for (string &strpair : strpair_cont) {
            string::size_type  delim_pos = strpair.rfind("_");
            assert(delim_pos != string::npos);
            std::string word = strpair.substr(0, delim_pos);
            // Parse Number to specific string
            word = replace_number(word);
            Index dynamic_word_id = dc_m.dynamic_dict_wrapper.Convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
            Index fixed_word_id = dc_m.fixed_dict.Convert(word);
            Index postag_id = dc_m.postag_dict.Convert(strpair.substr(delim_pos + 1));
            dynamic_sent.push_back(dynamic_word_id);
            fixed_sent.push_back(fixed_word_id);
            postag_seq.push_back(postag_id);
        }
        tmp_dynamic_sents.push_back(dynamic_sent);
        tmp_fixed_sents.push_back(fixed_sent);
        tmp_postag_seqs.push_back(postag_seq);
        ++line_cnt;
        if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << "lines"; }
    }
    swap(dynamic_sents, tmp_dynamic_sents);
    swap(fixed_sents, tmp_fixed_sents);
    swap(postag_seqs, tmp_postag_seqs);
}

void DoubleChannelModelHandler::read_training_data_and_build_dicts(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
    vector<IndexSeq> &postag_seqs)
{
    assert(!dc_m.dynamic_dict.is_frozen() && !dc_m.fixed_dict.is_frozen() && !dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, dynamic_sents, fixed_sents, postag_seqs);
}

void DoubleChannelModelHandler::read_devel_data(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
    vector<IndexSeq> &postag_seqs)
{
    assert(dc_m.dynamic_dict.is_frozen() &&!dc_m.fixed_dict.is_frozen() && dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, dynamic_sents, fixed_sents, postag_seqs);
}

void DoubleChannelModelHandler::read_test_data(istream &is, vector<Seq> &raw_test_sents, vector<IndexSeq> &dynamic_sents,
    vector<IndexSeq> &fixed_sents)
{
    assert(dc_m.dynamic_dict.is_frozen() && !dc_m.fixed_dict.is_frozen() && dc_m.postag_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read test data .";
    vector<Seq> tmp_raw_sents;
    vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents;
    string line;
    while (getline(is, line))
    {
        boost::trim(line);
        vector<string> words_seq;
        boost::split(words_seq, line, boost::is_any_of("\t"));
        tmp_raw_sents.push_back(words_seq);
        unsigned seq_len = words_seq.size();
        tmp_dynamic_sents.emplace_back(seq_len); // using constructor `vector(nr_num)` => push_back(vector<int>(nr_words)) 
        tmp_fixed_sents.emplace_back(seq_len);
        IndexSeq &dynamic_words_index_seq = tmp_dynamic_sents.back() ,
            &fixed_words_index_seq = tmp_fixed_sents.back();
        for (unsigned i = 0; i < seq_len; ++i)
        {
            string number_transed_word = replace_number(words_seq[i]);
            dynamic_words_index_seq[i] = dc_m.dynamic_dict.Convert(number_transed_word);
            fixed_words_index_seq[i] = dc_m.fixed_dict.Convert(number_transed_word);
        }

    }
    swap(tmp_raw_sents, raw_test_sents);
    swap(tmp_dynamic_sents, dynamic_sents);
    swap(tmp_fixed_sents, fixed_sents);
}

void DoubleChannelModelHandler::finish_read_training_data()
{
    dc_m.freeze_dict_and_add_UNK();
}

void DoubleChannelModelHandler::build_model(boost::program_options::variables_map &varmap)
{
    dc_m.set_partial_model_structure_param_from_outer(varmap);
    dc_m.set_partial_model_structure_param_from_inner();
    dc_m.build_model_structure();
    dc_m.print_model_info();
}

void DoubleChannelModelHandler::train(const vector<IndexSeq> *p_dynamic_sents, const vector<IndexSeq> *p_fixed_sents,
    const vector<IndexSeq> *p_postag_seqs,
    unsigned max_epoch,
    const vector<IndexSeq> *p_dev_dynamic_sents = nullptr, const vector<IndexSeq> *p_dev_fixed_sents = nullptr,
    const vector<IndexSeq> *p_dev_postag_seqs = nullptr,
    const unsigned long do_devel_freq = 50000)
{
    unsigned nr_samples = p_dynamic_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";

    vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    SimpleSGDTrainer sgd = SimpleSGDTrainer(dc_m.m);
    unsigned long line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq dynamic_sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *rndeng);

        // For loss , accuracy , time cost report
        Stat training_stat_per_report, training_stat_per_epoch;
        unsigned report_freq = 50000;

        // training for an epoch
        training_stat_per_report.start_time_stat();
        training_stat_per_epoch.start_time_stat();
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq *p_dynamic_sent = &p_dynamic_sents->at(access_idx),
                *p_fixed_sent = &p_fixed_sents->at(access_idx) ,
                *p_tag_seq = &p_postag_seqs->at(access_idx);
            ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                           // so we have to declaring it as pointer and destroy it handly 
                                                           // before develing.
                                                           // transform low-frequent words to UNK according to the probability
            dynamic_sent_after_replace_unk.resize(p_dynamic_sent->size());
            for (size_t word_idx = 0; word_idx < p_dynamic_sent->size(); ++word_idx)
            {
                dynamic_sent_after_replace_unk[word_idx] = dc_m.dynamic_dict_wrapper.ConvertProbability(p_dynamic_sent->at(word_idx));
            }
            dc_m.negative_loglikelihood(cg, &dynamic_sent_after_replace_unk, p_fixed_sent, p_tag_seq,&training_stat_per_report);
            training_stat_per_report.loss += as_scalar(cg->forward());
            cg->backward();
            sgd.update(1.0);
            delete cg;

            if (0 == (i + 1) % report_freq) // Report 
            {
                training_stat_per_report.end_time_stat();
                BOOST_LOG_TRIVIAL(trace) << i + 1 << " instances have been trained , with E = "
                    << training_stat_per_report.get_E()
                    << " , ACC = " << training_stat_per_report.get_acc() * 100
                    << " % with time cost " << training_stat_per_report.get_time_cost_in_seconds()
                    << " s .";
                training_stat_per_epoch += training_stat_per_report;
                training_stat_per_report.clear();
                training_stat_per_report.start_time_stat();
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if (p_dev_dynamic_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
            {
                float acc = devel(p_dev_dynamic_sents , p_dev_fixed_sents , p_dev_postag_seqs);
                if (acc > best_acc) save_current_best_model(acc);
                line_cnt_for_devel = 0; // avoid overflow
            }
        }

        // End of an epoch 
        sgd.update_epoch();

        training_stat_per_epoch.end_time_stat();
        training_stat_per_epoch += training_stat_per_report;

        // Output
        long long epoch_time_cost = training_stat_per_epoch.get_time_cost_in_seconds();
        BOOST_LOG_TRIVIAL(info) << "-------- epoch " << nr_epoch + 1 << "/" << max_epoch << " finished . ----------\n"
            << nr_samples << " instances has been trained .\n"
            << "For this epoch , E = " << training_stat_per_epoch.get_E() << "\n"
            << "ACC = " << training_stat_per_epoch.get_acc() * 100 << "  % \n"
            << "total time cost " << epoch_time_cost << " s ( speed "
            << training_stat_per_epoch.get_speed_as_kilo_tokens_per_sencond() << " k/s tokens).\n"
            << "total tags : " << training_stat_per_epoch.total_tags << " with correct tags : "
            << training_stat_per_epoch.correct_tags << " \n";

        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();

        // do validation at every ends of epoch
        if (p_dev_dynamic_sents != nullptr)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float acc = devel(p_dev_dynamic_sents , p_dev_fixed_sents , p_dev_postag_seqs);
            if (acc > best_acc) save_current_best_model(acc);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with cost " << total_time_cost_in_seconds << " s .";
}

float DoubleChannelModelHandler::devel(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
    const std::vector<IndexSeq> *p_postag_seqs,
    std::ostream *p_error_output_os = nullptr)
{
    unsigned nr_samples = p_dynamic_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .\n";
    unsigned long line_cnt4error_output = 0;
    if (p_error_output_os) *p_error_output_os << "line_nr\tword_index\tword_at_dict\tpredict_tag\ttrue_tag\n";
    Stat acc_stat;
    acc_stat.start_time_stat();
    for (int access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        ++line_cnt4error_output;
        ComputationGraph cg;
        IndexSeq predict_tag_seq;
        const IndexSeq *p_dynamic_sent = &p_dynamic_sents->at(access_idx),
            *p_fixed_sent = &p_fixed_sents->at(access_idx) ,
            *p_tag_seq = &p_postag_seqs->at(access_idx);
        dc_m.do_predict(&cg, p_dynamic_sent, p_fixed_sent, &predict_tag_seq);
        assert(predict_tag_seq.size() == p_tag_seq->size());
        for (unsigned i = 0; i < p_tag_seq->size(); ++i)
        {
            ++acc_stat.total_tags;
            if (p_tag_seq->at(i) == predict_tag_seq[i]) ++acc_stat.correct_tags;
            else if (p_error_output_os)
            {
                *p_error_output_os << line_cnt4error_output << "\t" << i << "\t" << dc_m.dynamic_dict.Convert(p_dynamic_sent->at(i))
                    << "\t" << dc_m.postag_dict.Convert(predict_tag_seq[i]) << "\t" << dc_m.postag_dict.Convert(p_tag_seq->at(i)) << "\n";
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

void DoubleChannelModelHandler::predict(std::istream &is, std::ostream &os)
{
    const string SPLIT_DELIMITER = "\t";
    vector<Seq> raw_instances;
    vector<IndexSeq> dynamic_sents,
        fixed_sents;
    read_test_data(is,raw_instances,dynamic_sents ,fixed_sents);
    for (unsigned int i = 0; i < raw_instances.size(); ++i)
    {
        vector<string> *p_raw_sent = &raw_instances.at(i);
        if (0 == p_raw_sent->size())
        {
            os << "\n";
            continue;
        }
        IndexSeq *p_dynamic_sent = &dynamic_sents.at(i) ,
            *p_fixed_sent = &fixed_sents.at(i);
        IndexSeq predict_seq;
        ComputationGraph cg;
        dc_m.do_predict(&cg, p_dynamic_sent, p_fixed_sent , &predict_seq);
        // output the result directly
        os << p_raw_sent->at(0) << "_" << dc_m.postag_dict.Convert(predict_seq.at(0));
        for (unsigned k = 1; k < p_raw_sent->size(); ++k)
        {
            os << SPLIT_DELIMITER
                << p_raw_sent->at(k) << "_" << dc_m.postag_dict.Convert(predict_seq.at(k));
        }
        os << "\n";
    }
}

void DoubleChannelModelHandler::save_model(std::ostream &os)
{
    dc_m.save_model(os, &best_model_tmp_ss);
}

void DoubleChannelModelHandler::load_model(std::istream &is)
{
    dc_m.load_model(is);
}

} // end of namespace