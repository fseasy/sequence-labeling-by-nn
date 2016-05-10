
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "utils/typedeclaration.h"
#include "ner_dc_modelhandler.h"

using namespace std;
using namespace cnn;
namespace slnn
{


const std::string NERDCModelHandler::number_transform_str = "##";
const size_t NERDCModelHandler::length_transform_str = number_transform_str.length();

NERDCModelHandler::NERDCModelHandler() 
    :dc_m(NERDCModel()) , best_F1(0.f) , best_model_tmp_ss()
{}

void NERDCModelHandler::build_fixed_dict_from_word2vec_file(std::ifstream &is)
{
    BOOST_LOG_TRIVIAL(info) << "initialize fixed dict .";
    string line;
    vector<string> split_cont;
    getline(is, line); // first line is the infomation !
    boost::split(split_cont, line, boost::is_any_of(" "));
    assert(2U == split_cont.size());
    dc_m.fixed_embedding_dict_size = stol(split_cont[0]) + 1; // another UNK
    dc_m.fixed_embedding_dim = stol(split_cont[1]);
    
    // read all words and add to dc_m.fixed_dict
    while (getline(is, line))
    {
        string::size_type delim_pos = line.find(" ");
        assert(delim_pos != string::npos);
        string word = line.substr(0, delim_pos);
        dc_m.fixed_dict.Convert(word);  // add to dict
    }
    //  freeze & add unk to fixed_dict
    dc_m.fixed_dict.Freeze();
    dc_m.fixed_dict.SetUnk(dc_m.UNK_STR);
    BOOST_LOG_TRIVIAL(info) << "initialize fixed dict done .";
}

void NERDCModelHandler::set_unk_replace_threshold(int freq_thres, float prob_thres)
{
    dc_m.dynamic_dict_wrapper.set_threshold(freq_thres, prob_thres);
}

void NERDCModelHandler::do_read_annotated_dataset(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
    vector<IndexSeq> &postag_seqs , vector<IndexSeq> &ner_seqs)
{
    unsigned line_cnt = 0;
    string line;
    vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents,
        tmp_postag_seqs,
        tmp_ner_seqs;
    IndexSeq dynamic_sent, 
        fixed_sent ,
        postag_seq,
        ner_seq;
    // pre-allocation
    tmp_dynamic_sents.reserve(MaxSentNum); // 2^19 =  480k pairs 
    tmp_fixed_sents.reserve(MaxSentNum);
    tmp_postag_seqs.reserve(MaxSentNum);
    tmp_ner_seqs.reserve(MaxSentNum);

    dynamic_sent.reserve(SentMaxLen);
    fixed_sent.reserve(SentMaxLen);
    postag_seq.reserve(SentMaxLen);
    ner_seq.reserve(SentMaxLen);

    while (getline(is, line)) {
        boost::algorithm::trim(line);
        if (0 == line.size()) continue;
        vector<string> parts;
        boost::algorithm::split(parts, line, boost::is_any_of("\t"));
        dynamic_sent.clear();
        fixed_sent.clear();
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
            Index dynamic_word_id = dc_m.dynamic_dict_wrapper.Convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
            Index fixed_word_id = dc_m.fixed_dict.Convert(word);
            Index postag_id = dc_m.postag_dict.Convert(postag);
            Index nertag_id = dc_m.ner_dict.Convert(nertag);
            dynamic_sent.push_back(dynamic_word_id);
            fixed_sent.push_back(fixed_word_id);
            postag_seq.push_back(postag_id);
            ner_seq.push_back(nertag_id);
        }
        tmp_dynamic_sents.push_back(dynamic_sent);
        tmp_fixed_sents.push_back(fixed_sent);
        tmp_postag_seqs.push_back(postag_seq);
        tmp_ner_seqs.push_back(ner_seq);
        ++line_cnt;
        if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    swap(dynamic_sents, tmp_dynamic_sents);
    swap(fixed_sents, tmp_fixed_sents);
    swap(postag_seqs, tmp_postag_seqs);
    swap(ner_seqs, tmp_ner_seqs);
}

void NERDCModelHandler::read_training_data_and_build_dicts(istream &is, vector<IndexSeq> &dynamic_sents, 
    vector<IndexSeq> &fixed_sents, vector<IndexSeq> &postag_seqs ,
    vector<IndexSeq> &ner_seqs)
{
    assert(dc_m.fixed_dict.is_frozen()); // fixed_dict should be Initialized .
    assert(!dc_m.dynamic_dict.is_frozen() && !dc_m.postag_dict.is_frozen() && !dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read training data .";
    do_read_annotated_dataset(is, dynamic_sents, fixed_sents, postag_seqs , ner_seqs);
    dc_m.dynamic_dict_wrapper.Freeze();
    dc_m.dynamic_dict_wrapper.SetUnk(dc_m.UNK_STR);
    dc_m.postag_dict.Freeze();
    dc_m.ner_dict.Freeze();
    BOOST_LOG_TRIVIAL(info) << "read training data done and set dynamic , postag and ner dict done . ";
}

void NERDCModelHandler::read_devel_data(istream &is, vector<IndexSeq> &dynamic_sents, vector<IndexSeq> &fixed_sents,
    vector<IndexSeq> &postag_seqs , vector<IndexSeq> &ner_seqs)
{
    assert(dc_m.dynamic_dict.is_frozen() && dc_m.fixed_dict.is_frozen() && dc_m.postag_dict.is_frozen()
           && dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read developing data .";
    do_read_annotated_dataset(is, dynamic_sents, fixed_sents, postag_seqs , ner_seqs);
    BOOST_LOG_TRIVIAL(info) << "read developing data done .";
}

void NERDCModelHandler::read_test_data(istream &is, vector<Seq> &raw_test_sents, vector<IndexSeq> &dynamic_sents,
    vector<IndexSeq> &fixed_sents , vector<IndexSeq> &postag_seqs)
{
    assert(dc_m.dynamic_dict.is_frozen() && dc_m.fixed_dict.is_frozen() && 
        dc_m.postag_dict.is_frozen() && dc_m.ner_dict.is_frozen());
    BOOST_LOG_TRIVIAL(info) << "read test data .";
    vector<Seq> tmp_raw_sents;
    vector<IndexSeq> tmp_dynamic_sents,
        tmp_fixed_sents , 
        tmp_postag_seqs;
    string line;
    while (getline(is, line))
    {
        boost::trim(line);
        vector<string> parts_seq;
        boost::split(parts_seq, line, boost::is_any_of("\t"));
        unsigned seq_len = parts_seq.size();
        tmp_raw_sents.emplace_back(seq_len);
        tmp_dynamic_sents.emplace_back(seq_len); // using constructor `vector(nr_num)` => push_back(vector<int>(nr_words)) 
        tmp_fixed_sents.emplace_back(seq_len);
        tmp_postag_seqs.emplace_back(seq_len);
        Seq &raw_sent = tmp_raw_sents.back();
        IndexSeq &dynamic_words_index_seq = tmp_dynamic_sents.back() ,
            &fixed_words_index_seq = tmp_fixed_sents.back() , 
            &postag_index_seq = tmp_postag_seqs.back();
        for (unsigned i = 0; i < seq_len; ++i)
        {
            string &part = parts_seq.at(i);
            string::size_type delim_pos = part.rfind("_");
            string raw_word = part.substr(0, delim_pos);
            string postag = part.substr(delim_pos + 1);
            string number_transed_word = replace_number(raw_word);
            raw_sent[i] = raw_word ;
            dynamic_words_index_seq[i] = dc_m.dynamic_dict.Convert(number_transed_word);
            fixed_words_index_seq[i] = dc_m.fixed_dict.Convert(number_transed_word);
            postag_index_seq[i] = dc_m.postag_dict.Convert(postag);
        }

    }
    swap(tmp_raw_sents, raw_test_sents);
    swap(tmp_dynamic_sents, dynamic_sents);
    swap(tmp_fixed_sents, fixed_sents);
    swap(tmp_postag_seqs, postag_seqs);
}

void NERDCModelHandler::finish_read_training_data(boost::program_options::variables_map &varmap)
{
    assert(dc_m.dynamic_dict.is_frozen() && dc_m.fixed_dict.is_frozen() && dc_m.postag_dict.is_frozen()
        && dc_m.ner_dict.is_frozen());
    // set param 
    dc_m.dynamic_embedding_dim = varmap["dynamic_embedding_dim"].as<unsigned>();
    dc_m.postag_embedding_dim = varmap["postag_embedding_dim"].as<unsigned>();
    dc_m.ner_embedding_dim = varmap["ner_embedding_dim"].as<unsigned>();
    dc_m.nr_lstm_stacked_layer = varmap["nr_lstm_stacked_layer"].as<unsigned>();
    dc_m.lstm_x_dim = varmap["lstm_x_dim"].as<unsigned>();
    dc_m.lstm_h_dim = varmap["lstm_h_dim"].as<unsigned>();
    dc_m.tag_layer_hidden_dim = varmap["tag_layer_hidden_dim"].as<unsigned>();

    dc_m.dynamic_embedding_dict_size = dc_m.dynamic_dict.size();
    dc_m.postag_embedding_dict_size = dc_m.postag_dict.size();
    dc_m.ner_embedding_dict_size = dc_m.ner_dict.size();
    assert(dc_m.fixed_embedding_dict_size == dc_m.fixed_dict.size());
}

void NERDCModelHandler::build_model()
{
    dc_m.build_model_structure();
    dc_m.print_model_info();
}

void NERDCModelHandler::load_fixed_embedding(std::istream &is)
{
    // set lookup parameters from outer word embedding
    // using words_loopup_param.Initialize( word_id , value_vector )
    BOOST_LOG_TRIVIAL(info) << "load pre-trained word embedding .";
    string line;
    vector<string> split_cont;
    getline(is, line); // first line is the infomation , skip
    split_cont.reserve(dc_m.fixed_embedding_dim + 1); // word + numbers 
    unsigned long line_cnt = 0; // for warning when read embedding error
    unsigned long words_cnt_hit = 0;
    vector<float> embedding_vec(dc_m.fixed_embedding_dim , 0.f);
    Index dynamic_unk = dc_m.dynamic_dict.Convert(dc_m.UNK_STR); // for calc hit rate
    while (getline(is, line))
    {
        ++line_cnt;
        boost::trim_right(line);
        boost::split(split_cont, line, boost::is_any_of(" "));
        if (dc_m.fixed_embedding_dim + 1 != split_cont.size())
        {
            BOOST_LOG_TRIVIAL(info) << "bad word dimension : `" << split_cont.size() - 1 << "` at line " << line_cnt;
            continue;
        }
        string &word = split_cont.at(0);
        Index word_id = dc_m.fixed_dict.Convert(word);
        for (size_t idx = 1; idx < split_cont.size(); ++idx)
        {
            embedding_vec[idx - 1] = stof(split_cont[idx]);
        }
        dc_m.fixed_words_lookup_param->Initialize(word_id, embedding_vec);
        if(dc_m.dynamic_dict.Convert(word) != dynamic_unk) ++words_cnt_hit;
    }
    BOOST_LOG_TRIVIAL(info) << "load fixed embedding done . hit rate " 
        << words_cnt_hit << "/" << dc_m.fixed_embedding_dict_size  << " ("
        << ( dc_m.fixed_embedding_dict_size ? static_cast<float>(words_cnt_hit) / dc_m.fixed_embedding_dict_size : 0. ) * 100 
        << " %) " ;
}

void NERDCModelHandler::train(const vector<IndexSeq> *p_dynamic_sents, const vector<IndexSeq> *p_fixed_sents,
    const vector<IndexSeq> *p_postag_seqs, const vector<IndexSeq> *p_ner_seqs ,
    unsigned max_epoch,
    const vector<IndexSeq> *p_dev_dynamic_sents, const vector<IndexSeq> *p_dev_fixed_sents,
    const vector<IndexSeq> *p_dev_postag_seqs, const vector<IndexSeq> *p_dev_ner_seqs ,
    const string &conlleval_script_path , 
    unsigned do_devel_freq ,
    unsigned trivial_report_freq)
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

        // training for an epoch
        training_stat_per_report.start_time_stat();
        training_stat_per_epoch.start_time_stat();
        for (unsigned i = 0; i < nr_samples; ++i)
        {
            unsigned access_idx = access_order[i];
            // using negative_loglikelihood loss to build model
            const IndexSeq *p_dynamic_sent = &p_dynamic_sents->at(access_idx),
                *p_fixed_sent = &p_fixed_sents->at(access_idx) ,
                *p_postag_seq = &p_postag_seqs->at(access_idx) ,
                *p_ner_seq = &p_ner_seqs->at(access_idx);
            ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                           // so we have to declaring it as pointer and destroy it handly 
                                                           // before develing.
                                                           // transform low-frequent words to UNK according to the probability
            dynamic_sent_after_replace_unk.resize(p_dynamic_sent->size());
            for (size_t word_idx = 0; word_idx < p_dynamic_sent->size(); ++word_idx)
            {
                dynamic_sent_after_replace_unk[word_idx] = dc_m.dynamic_dict_wrapper.ConvertProbability(p_dynamic_sent->at(word_idx));
            }
            dc_m.negative_loglikelihood(cg, &dynamic_sent_after_replace_unk, p_fixed_sent, 
                                        p_postag_seq, p_ner_seq ,  &training_stat_per_report);
            training_stat_per_report.loss += as_scalar(cg->forward());
            cg->backward();
            sgd.update(1.0);
            delete cg;

            if (0 == (i + 1) % trivial_report_freq) // Report 
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
                float F1 = devel(p_dev_dynamic_sents , p_dev_fixed_sents , p_dev_postag_seqs , 
                    p_dev_ner_seqs , conlleval_script_path);
                if (F1 > best_F1) save_current_best_model(F1);
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
            float F1 = devel(p_dev_dynamic_sents , p_dev_fixed_sents , p_dev_postag_seqs , 
                p_dev_ner_seqs , conlleval_script_path);
            if (F1 > best_F1) save_current_best_model(F1);
        }

    }
    BOOST_LOG_TRIVIAL(info) << "training finished with cost " << total_time_cost_in_seconds << " s .";
}

float NERDCModelHandler::devel(const std::vector<IndexSeq> *p_dynamic_sents, const std::vector<IndexSeq> *p_fixed_sents,
    const std::vector<IndexSeq> *p_postag_seqs, const vector<IndexSeq> *p_ner_seqs , 
    const string &conlleval_script_path)
{
    unsigned nr_samples = p_dynamic_sents->size();
    BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .\n";
    unsigned long line_cnt4error_output = 0;

    NerStat stat(conlleval_script_path);
    stat.start_time_stat();
    vector<IndexSeq> predict_ner_seqs(p_ner_seqs->size());
    for (unsigned access_idx = 0; access_idx < nr_samples; ++access_idx)
    {
        ++line_cnt4error_output;
        ComputationGraph cg;
        IndexSeq predict_ner_seq;
        const IndexSeq *p_dynamic_sent = &p_dynamic_sents->at(access_idx),
            *p_fixed_sent = &p_fixed_sents->at(access_idx) ,
            *p_postag_seq = &p_postag_seqs->at(access_idx);
        dc_m.do_predict(&cg, p_dynamic_sent, p_fixed_sent, p_postag_seq , &predict_ner_seq);
        predict_ner_seqs[access_idx] = predict_ner_seq;
        stat.total_tags += predict_ner_seq.size();
    }
    stat.end_time_stat();
    array<float , 4> eval_scores = stat.conlleval(*p_ner_seqs, predict_ner_seqs, dc_m.ner_dict);
    float Acc = eval_scores[0] , 
          P = eval_scores[1] ,
          R = eval_scores[2] ,
          F1 = eval_scores[3] ;
    BOOST_LOG_TRIVIAL(info) << "validation finished .\n"
        << "Acc = " << Acc << "% , P = " << P << "% , R = " << R << "% , F1 = "<< F1 << "%\n"
        << "Time cosing " << stat.get_time_cost_in_seconds() << " s . \n"
        << "Speed " << stat.get_speed_as_kilo_tokens_per_sencond() << " K tokens / s";
    return F1;
}

void NERDCModelHandler::predict(std::istream &is, std::ostream &os)
{
    const string SPLIT_DELIMITER = "\t";
    vector<Seq> raw_instances;
    vector<IndexSeq> dynamic_sents,
        fixed_sents , 
        postag_seqs ;
    read_test_data(is,raw_instances,dynamic_sents ,fixed_sents , postag_seqs);
    
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
        IndexSeq *p_dynamic_sent = &dynamic_sents.at(i) ,
            *p_fixed_sent = &fixed_sents.at(i) ,
            *p_postag_seq = &postag_seqs.at(i);
        IndexSeq predict_seq;
        ComputationGraph cg;
        dc_m.do_predict(&cg, p_dynamic_sent, p_fixed_sent , p_postag_seq , &predict_seq);
        // output the result directly
        os << p_raw_sent->at(0)
            << "/" << dc_m.postag_dict.Convert(p_postag_seq->at(0))
            << "#" << dc_m.ner_dict.Convert(predict_seq.at(0));
        for (unsigned k = 1; k < p_raw_sent->size(); ++k)
        {
            os << SPLIT_DELIMITER
                << p_raw_sent->at(k)
                << "/" << dc_m.postag_dict.Convert(p_postag_seq->at(k))
                << "#" << dc_m.ner_dict.Convert(predict_seq.at(k));
        }
        os << "\n";
        stat.total_tags += predict_seq.size() ;
    }
    stat.end_time_stat() ;
    BOOST_LOG_TRIVIAL(info) << "predict done . time cosing " << stat.get_time_cost_in_seconds() << " s , speed "
        << stat.get_speed_as_kilo_tokens_per_sencond() << " K tokens/s"  ;
}

void NERDCModelHandler::save_model(std::ostream &os)
{
    BOOST_LOG_TRIVIAL(info) << "saving model ...";
    boost::archive::text_oarchive to(os);
    to << dc_m.dynamic_embedding_dim << dc_m.fixed_embedding_dim
        << dc_m.postag_embedding_dim << dc_m.ner_embedding_dim
        << dc_m.nr_lstm_stacked_layer << dc_m.lstm_x_dim << dc_m.lstm_h_dim
        << dc_m.tag_layer_hidden_dim 
        << dc_m.dynamic_embedding_dict_size << dc_m.fixed_embedding_dict_size
        << dc_m.postag_embedding_dict_size << dc_m.ner_embedding_dict_size ;

    to << dc_m.dynamic_dict << dc_m.fixed_dict << dc_m.postag_dict << dc_m.ner_dict ;
    if (best_model_tmp_ss && 0 != best_model_tmp_ss.rdbuf()->in_avail())
    {
        boost::archive::text_iarchive ti(best_model_tmp_ss);
        ti >> *dc_m.m;
    }
    to << *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

void NERDCModelHandler::load_model(std::istream &is)
{
    BOOST_LOG_TRIVIAL(info) << "loading model ...";
    boost::archive::text_iarchive ti(is);
    ti >> dc_m.dynamic_embedding_dim >> dc_m.fixed_embedding_dim
        >> dc_m.postag_embedding_dim >> dc_m.ner_embedding_dim 
        >> dc_m.nr_lstm_stacked_layer >> dc_m.lstm_x_dim >> dc_m.lstm_h_dim
        >> dc_m.tag_layer_hidden_dim 
        >> dc_m.dynamic_embedding_dict_size >> dc_m.fixed_embedding_dict_size
        >> dc_m.postag_embedding_dict_size >> dc_m.ner_embedding_dict_size ;

    ti >> dc_m.dynamic_dict >> dc_m.fixed_dict >> dc_m.postag_dict >> dc_m.ner_dict ;
    assert(dc_m.dynamic_embedding_dict_size == dc_m.dynamic_dict.size() && dc_m.fixed_embedding_dict_size == dc_m.fixed_dict.size()
        && dc_m.postag_embedding_dict_size == dc_m.postag_dict.size() && dc_m.ner_embedding_dict_size == dc_m.ner_dict.size());
    build_model();
    ti >> *dc_m.m;
    BOOST_LOG_TRIVIAL(info) << "load model done .";
}

} // end of namespace
