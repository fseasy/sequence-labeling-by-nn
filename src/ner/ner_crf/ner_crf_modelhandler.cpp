
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "utils/typedeclaration.h"
#include "ner_crf_modelhandler.h"

using namespace std;
using namespace dynet;
namespace slnn
{


const std::string NERCRFModelHandler::number_transform_str = "##";
const size_t NERCRFModelHandler::length_transform_str = number_transform_str.length();

NERCRFModelHandler::NERCRFModelHandler() 
    :dc_m(NERCRFModel()) , best_F1(0.f) , best_model_tmp_ss()
{}

void NERCRFModelHandler::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    dc_m.word_dict_wrapper.set_threshold(freq_thres, prob_thres);
}

void NERCRFModelHandler::do_read_annotated_dataset(istream &is, vector<IndexSeq> &sents,
    vector<IndexSeq> &postag_seqs , vector<IndexSeq> &ner_seqs)
{
    unsigned line_cnt = 0;
    string line;
    vector<IndexSeq> tmp_sents,
        tmp_postag_seqs,
        tmp_ner_seqs;
    IndexSeq sent, 
        postag_seq,
        ner_seq;
    // pre-allocation
    tmp_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_postag_seqs.reserve(MaxSentNum);
    tmp_ner_seqs.reserve(MaxSentNum);

    sent.reserve(SentMaxLen);
    postag_seq.reserve(SentMaxLen);
    ner_seq.reserve(SentMaxLen);

    while (getline(is, line)) {
        boost::algorithm::trim(line);
        if (0 == line.size()) continue;
        vector<string> parts;
        boost::algorithm::split(parts, line, boost::is_any_of("\t"));
        sent.clear();
        postag_seq.clear();
        ner_seq.clear();
        for (string &part : parts) {
            string::size_type postag_pos = part.rfind("/");
            string::size_type nertag_pos = part.rfind("#");
            assert(postag_pos != string::npos && nertag_pos != string::npos);
            string word = part.substr(0, postag_pos);
            string postag = part.substr(postag_pos + 1, nertag_pos - postag_pos - 1);
            string nertag = part.substr(nertag_pos + 1);
            // Parse Number to specific string
            word = replace_number(word);
            Index word_id = dc_m.word_dict_wrapper.convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
            Index postag_id = dc_m.postag_dict.convert(postag);
            Index nertag_id = dc_m.ner_dict.convert(nertag);
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

void NERCRFModelHandler::read_training_data_and_build_dicts(istream &is, vector<IndexSeq> &sents, 
    vector<IndexSeq> &postag_seqs ,
    vector<IndexSeq> &ner_seqs)
{
    assert(!dc_m.word_dict.is_frozen() && !dc_m.postag_dict.is_frozen() && !dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, sents,  postag_seqs , ner_seqs);
    dc_m.word_dict_wrapper.freeze();
    dc_m.word_dict_wrapper.set_unk(dc_m.UNK_STR);
    dc_m.postag_dict.freeze();
    dc_m.ner_dict.freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set dynamic , postag and ner dict done . ";
}

void NERCRFModelHandler::read_devel_data(istream &is, vector<IndexSeq> &sents,
    vector<IndexSeq> &postag_seqs , vector<IndexSeq> &ner_seqs)
{
    assert(dc_m.word_dict.is_frozen() && dc_m.postag_dict.is_frozen()
           && dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, sents, postag_seqs , ner_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

void NERCRFModelHandler::read_test_data(istream &is, vector<Seq> &raw_test_sents, vector<IndexSeq> &sents,
    vector<IndexSeq> &postag_seqs)
{
    assert(dc_m.word_dict.is_frozen() && 
        dc_m.postag_dict.is_frozen() && dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read test data .";
    vector<Seq> tmp_raw_sents;
    vector<IndexSeq> tmp_sents,
        tmp_postag_seqs;
    string line;
    while (getline(is, line))
    {
        boost::trim(line);
        vector<string> parts_seq;
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
            string &part = parts_seq.at(i);
            string::size_type delim_pos = part.rfind("_");
            string raw_word = part.substr(0, delim_pos);
            string postag = part.substr(delim_pos + 1);
            string number_transed_word = replace_number(raw_word);
            raw_sent[i] = raw_word ;
            words_index_seq[i] = dc_m.word_dict.convert(number_transed_word);
            postag_index_seq[i] = dc_m.postag_dict.convert(postag);
        }

    }
    swap(tmp_raw_sents, raw_test_sents);
    swap(tmp_sents, sents);
    swap(tmp_postag_seqs, postag_seqs);
}

void NERCRFModelHandler::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    assert(dc_m.word_dict.is_frozen() && dc_m.postag_dict.is_frozen()
        && dc_m.ner_dict.is_frozen());
    // set param 
    dc_m.word_embedding_dim = varmap["word_embedding_dim"].as<unsigned>();
    dc_m.postag_embedding_dim = varmap["postag_embedding_dim"].as<unsigned>();
    dc_m.ner_embedding_dim = varmap["ner_embedding_dim"].as<unsigned>();
    dc_m.nr_lstm_stacked_layer = varmap["nr_lstm_stacked_layer"].as<unsigned>();
    dc_m.lstm_x_dim = varmap["lstm_x_dim"].as<unsigned>();
    dc_m.lstm_h_dim = varmap["lstm_h_dim"].as<unsigned>();
    dc_m.emit_hidden_layer_dim = varmap["emit_hidden_layer_dim"].as<unsigned>();

    dc_m.word_embedding_dict_size = dc_m.word_dict.size();
    dc_m.postag_embedding_dict_size = dc_m.postag_dict.size();
    dc_m.ner_embedding_dict_size = dc_m.ner_dict.size();
}

void NERCRFModelHandler::build_model()
{
    dc_m.build_model_structure();
    dc_m.print_model_info();
}

void NERCRFModelHandler::train(const vector<IndexSeq> *p_sents, 
    const vector<IndexSeq> *p_postag_seqs, const vector<IndexSeq> *p_ner_seqs ,
    unsigned max_epoch,
    float dropout_rate , 
    const vector<IndexSeq> *p_dev_sents, 
    const vector<IndexSeq> *p_dev_postag_seqs, const vector<IndexSeq> *p_dev_ner_seqs ,
    const string &conlleval_script_path , 
    unsigned do_devel_freq ,
    bool do_stat_in_training , 
    unsigned trivial_report_freq)
{
    unsigned nr_samples = p_sents->size();

    BOOST_LOG_TRIVIAL(info) << "train at " << nr_samples << " instances .\n";

    vector<unsigned> access_order(nr_samples);
    for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

    SimpleSGDTrainer sgd = SimpleSGDTrainer(dc_m.m);
    unsigned line_cnt_for_devel = 0;
    unsigned long long total_time_cost_in_seconds = 0ULL;
    IndexSeq dynamic_sent_after_replace_unk(SentMaxLen , 0);
    for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
    {
        BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << "/" << max_epoch << " for train ";
        // shuffle samples by random access order
        shuffle(access_order.begin(), access_order.end(), *rndeng);

        // For loss , accuracy , time cost report
        Stat training_stat_per_epoch;
        shared_ptr<Stat> training_stat4trivial(nullptr);
        if (do_stat_in_training) training_stat4trivial = make_shared<Stat>();
        training_stat_per_epoch.start_time_stat();
        if (training_stat4trivial) training_stat4trivial->start_time_stat();

        // train for every Epoch 
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq *p_sent = &p_sents->at(access_idx),
                *p_postag_seq = &p_postag_seqs->at(access_idx) ,
                *p_ner_seq = &p_ner_seqs->at(access_idx);
            ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                           // so we have to declaring it as pointer and destroy it handly 
                                                           // before develing.
                                                           // transform low-frequent words to UNK according to the probability
            dynamic_sent_after_replace_unk.resize(p_sent->size());
            for (size_t word_idx = 0; word_idx < p_sent->size(); ++word_idx)
            {
                dynamic_sent_after_replace_unk[word_idx] = 
                    dc_m.word_dict_wrapper.unk_replace_probability(p_sent->at(word_idx));
            }
            auto loss_expr = dc_m.viterbi_train(cg, &dynamic_sent_after_replace_unk,
                                        p_postag_seq, p_ner_seq , dropout_rate ,  training_stat4trivial.get());
            dynet::real loss =  as_scalar(cg->forward(loss_expr));
            cg->backward(loss_expr);
            sgd.update(1.f);
            delete cg;
            if (training_stat4trivial) training_stat4trivial->loss += loss;
            else
            { 
                training_stat_per_epoch.loss += loss;
                training_stat_per_epoch.total_tags += p_sent->size() ;
            }
            if (0 == (i + 1) % trivial_report_freq) // Report 
            {
                string trivial_header = to_string(i + 1) + " instances have been trained.";
                if (do_stat_in_training)
                {
                    training_stat4trivial->end_time_stat();
                    BOOST_LOG_TRIVIAL(trace) << training_stat4trivial->get_stat_str(trivial_header );
                    training_stat_per_epoch += (*training_stat4trivial);
                    training_stat4trivial->clear();
                    training_stat4trivial->start_time_stat();
                }
                else
                {
                    BOOST_LOG_TRIVIAL(trace) << training_stat_per_epoch.get_basic_stat_str(trivial_header);
                }
            }

            // Devel
            ++line_cnt_for_devel;
            // If developing samples is available , do `devel` to get model training effect . 
            if (p_dev_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
            {
                float F1 = devel(p_dev_sents  , p_dev_postag_seqs , 
                    p_dev_ner_seqs , conlleval_script_path);
                if (F1 > best_F1) save_current_best_model(F1);
                line_cnt_for_devel = 0; // avoid overflow
            }
        }

        // End of an epoch 
        sgd.update_epoch();

        training_stat_per_epoch.end_time_stat();
        if(do_stat_in_training) training_stat_per_epoch += *training_stat4trivial;

        // Output at end of every eopch
        ostringstream tmp_sos;
        tmp_sos << "-------- epoch " << nr_epoch + 1 << "/" << to_string(max_epoch) << " finished . ----------\n"
            << nr_samples << " instances has been trained . \n";
        string info_header = tmp_sos.str();
        if (do_stat_in_training)
        {
            BOOST_LOG_TRIVIAL(info) << training_stat_per_epoch.get_stat_str(info_header);
        }
        else
        {
            BOOST_LOG_TRIVIAL(info) << training_stat_per_epoch.get_basic_stat_str(info_header);
        }
        total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
        // do validation at every ends of epoch
        if (p_dev_sents != nullptr)
        {
            BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch .";
            float F1 = devel(p_dev_sents , p_dev_postag_seqs , 
                p_dev_ner_seqs , conlleval_script_path);
            if (F1 > best_F1) save_current_best_model(F1);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with time cost " << total_time_cost_in_seconds << " s .";
}

float NERCRFModelHandler::devel(const std::vector<IndexSeq> *p_sents,
    const std::vector<IndexSeq> *p_postag_seqs, const vector<IndexSeq> *p_ner_seqs , 
    const string &conlleval_script_path)
{
    unsigned nr_samples = p_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";

    NerStat stat(conlleval_script_path);
    stat.start_time_stat();
    vector<IndexSeq> predict_ner_seqs(p_ner_seqs->size());
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        ComputationGraph cg;
        IndexSeq predict_ner_seq;
        const IndexSeq *p_sent = &p_sents->at(access_idx),
            *p_postag_seq = &p_postag_seqs->at(access_idx);
        dc_m.viterbi_predict(&cg, p_sent,p_postag_seq , &predict_ner_seq);
        predict_ner_seqs[access_idx] = predict_ner_seq;
        stat.total_tags += predict_ner_seq.size();
    }
    stat.end_time_stat();
    array<float , 4> eval_scores = stat.conlleval(*p_ner_seqs, predict_ner_seqs, dc_m.ner_dict);
    float Acc = eval_scores[0] , 
          P = eval_scores[1] ,
          R = eval_scores[2] ,
          F1 = eval_scores[3] ;
    ostringstream tmp_sos;
    tmp_sos << "validation finished .\n"
        << "Acc = " << Acc << "% , P = " << P << "% , R = " << R << "% , F1 = " << F1 << "%";
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str(tmp_sos.str()) ;
    return F1;
}

void NERCRFModelHandler::predict(std::istream &is, std::ostream &os)
{
    const string SPLIT_DELIMITER = "\t";
    vector<Seq> raw_instances;
    vector<IndexSeq> sents,
        postag_seqs ;
    read_test_data(is,raw_instances, sents , postag_seqs);
    
    BOOST_LOG_TRIVIAL(info) << "do prediction on " << raw_instances.size() << " instances .";
    BasicStat stat;
    stat.start_time_stat();
    for (unsigned int i = 0; i < raw_instances.size(); ++i)
    {
        vector<string> *p_raw_sent = &raw_instances.at(i);
        if (0 == p_raw_sent->size())
        {
            os << "\n";
            continue;
        }
        IndexSeq *p_sent = &sents.at(i) ,
            *p_postag_seq = &postag_seqs.at(i);
        IndexSeq predict_seq;
        ComputationGraph cg;
        dc_m.viterbi_predict(&cg, p_sent, p_postag_seq , &predict_seq);
        // output the result directly
        os << p_raw_sent->at(0)
            << "/" << dc_m.postag_dict.convert(p_postag_seq->at(0))
            << "#" << dc_m.ner_dict.convert(predict_seq.at(0));
        for (unsigned k = 1; k < p_raw_sent->size(); ++k)
        {
            os << SPLIT_DELIMITER
                << p_raw_sent->at(k)
                << "/" << dc_m.postag_dict.convert(p_postag_seq->at(k))
                << "#" << dc_m.ner_dict.convert(predict_seq.at(k));
        }
        os << "\n";
        stat.total_tags += predict_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << stat.get_stat_str("predict done.")  ;
}

void NERCRFModelHandler::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    boost::archive::text_oarchive to(os);
    to << dc_m.word_embedding_dim 
        << dc_m.postag_embedding_dim << dc_m.ner_embedding_dim
        << dc_m.nr_lstm_stacked_layer << dc_m.lstm_x_dim << dc_m.lstm_h_dim
        << dc_m.emit_hidden_layer_dim 
        << dc_m.word_embedding_dict_size 
        << dc_m.postag_embedding_dict_size << dc_m.ner_embedding_dict_size ;

    to << dc_m.word_dict << dc_m.postag_dict << dc_m.ner_dict ;
    if (best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail())
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *dc_m.m;
    }
    to << *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

void NERCRFModelHandler::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is);
    ti >> dc_m.word_embedding_dim 
        >> dc_m.postag_embedding_dim >> dc_m.ner_embedding_dim 
        >> dc_m.nr_lstm_stacked_layer >> dc_m.lstm_x_dim >> dc_m.lstm_h_dim
        >> dc_m.emit_hidden_layer_dim
        >> dc_m.word_embedding_dict_size
        >> dc_m.postag_embedding_dict_size >> dc_m.ner_embedding_dict_size ;

    ti >> dc_m.word_dict  >> dc_m.postag_dict >> dc_m.ner_dict ;
    assert(dc_m.word_embedding_dict_size == dc_m.word_dict.size()
        && dc_m.postag_embedding_dict_size == dc_m.postag_dict.size() && dc_m.ner_embedding_dict_size == dc_m.ner_dict.size());
    build_model();
    ti >> *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "load model done .";
}

} // end of namespace
