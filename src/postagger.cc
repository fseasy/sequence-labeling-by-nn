#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace cnn;
namespace po = boost::program_options;

using Index = int; // cnn::Dict return `int` as index 
using IndexSeq = vector<Index>;
using InstancePair = pair<IndexSeq, IndexSeq>;

// HPC PATH
//const string POS_TRAIN_PATH = "/data/ltp/ltp-data/pos/pku-weibo-train.pos" ;
//const string POS_DEV_PATH = "/data/ltp/ltp-data/pos/pku-weibo-holdout.pos" ;
//const string POS_TEST_PATH = "/data/ltp/ltp-data/pos/pku-weibo-test.pos" ;

// WINDOWS LOCAL
const string POS_TRAIN_PATH = "C:/data/ltp-data/ner/pku-weibo-train.pos";
const string POS_DEV_PATH = "C:/data/ltp-data/ner/pku-weibo-holdout.pos";
const string POS_TEST_PATH = "C:/data/ltp-data/ner/pku-weibo-test.pos";



template<typename Iterator>
void print(Iterator begin, Iterator end)
{
    for (Iterator i = begin; i != end; ++i)
    {
        cout << *i << "\t";
    }
    cout << endl;
}



void print_instance_pair(const vector<InstancePair> &cont, const cnn::Dict &word_dict, const cnn::Dict &tag_dict)
{
    for (const InstancePair & instance_pair : cont)
    {
        const IndexSeq &sent_indices = instance_pair.first,
            &tag_indices = instance_pair.second;
        for (auto sent_id : sent_indices) cout << word_dict.Convert(sent_id) << " ";
        cout << "\n";
        for (auto tag_id : tag_indices) cout << tag_dict.Convert(tag_id) << " ";
        cout << "\n" << endl;
    }
}


struct Stat
{
    unsigned long correct_tags;
    unsigned long total_tags;
    float loss;
    chrono::high_resolution_clock::time_point time_start;
    chrono::high_resolution_clock::time_point time_end;
    Stat() :correct_tags(0), total_tags(0), loss(0) {};
    float get_acc() { return total_tags != 0 ? float(correct_tags) / total_tags : 0.f; }
    float get_E() { return total_tags != 0 ? loss / total_tags : 0.f; }
    void clear() { correct_tags = 0; total_tags = 0; loss = 0.f; }
    chrono::high_resolution_clock::time_point start_time_stat() { return time_start = chrono::high_resolution_clock::now(); }
    chrono::high_resolution_clock::time_point end_time_stat() { return time_end = chrono::high_resolution_clock::now(); }
    double get_time_cost_in_seconds() { return (chrono::duration_cast<chrono::duration<double>>(time_end - time_start)).count(); }
    Stat &operator+=(const Stat &other)
    {
        correct_tags += other.correct_tags;
        total_tags += other.total_tags;
        loss += other.loss;
        time_end = other.time_end;
        return *this;
    }
    Stat operator+(const Stat &other) { Stat tmp = *this;  tmp += other;  return tmp; }
};


struct BILSTMModel4Tagging
{
    // model 
    Model *m;
    LSTMBuilder* l2r_builder;
    LSTMBuilder* r2l_builder;

    // paramerters
    LookupParameters *words_lookup_param;
    Parameters *l2r_tag_hidden_w_param;
    Parameters *r2l_tag_hidden_w_param;
    Parameters *tag_hidden_b_param;

    Parameters *tag_output_w_param;
    Parameters *tag_output_b_param;

    // model structure : using const !(in-class initilization)

    unsigned INPUT_DIM; // word embedding dimension
    unsigned LSTM_LAYER;
    unsigned LSTM_HIDDEN_DIM;
    unsigned TAG_HIDDEN_DIM;
    //--- need to be counted from training data or loaded from model .
    unsigned TAG_OUTPUT_DIM;
    unsigned WORD_DICT_SIZE;

    // others 
    cnn::Dict word_dict;
    cnn::Dict tag_dict;
    const string SOS_STR = "<START_OF_SEQUENCE_REPR>";
    const string EOS_STR = "<END_OF_SEQUENCE_REPR>";
    const string UNK_STR = "<UNK_REPR>"; // should add a unknown token into the lexicon. 
    Index SOS;
    Index EOS;
    Index UNK;

    BILSTMModel4Tagging() :
        m(nullptr), l2r_builder(nullptr), r2l_builder(nullptr)
    {

        SOS = word_dict.Convert(SOS_STR);
        EOS = word_dict.Convert(EOS_STR);
        // UNK is set after word dict is freezen
    }

    ~BILSTMModel4Tagging()
    {
        if (m) delete m;
        if (l2r_builder) delete l2r_builder;
        if (r2l_builder) delete r2l_builder;
    }

    /*************************READING DATA **********************************/

    void do_read_dataset(istream &is, vector<InstancePair> &samples)
    {
        // read training data or developing data .
        // data format :
        // Each line is combined with `WORD_TAG`s and delimeters , delimeter is only TAB . `WORD_TAG` should be split to `WORD` and `TAG`
        // Attention : empty line will be skipped 
        unsigned line_cnt = 0;
        string line;
        vector<InstancePair> tmp_samples;
        IndexSeq sent, tag_seq;
        tmp_samples.reserve(0x8FFFF); // 2^19 =  480k pairs 
        sent.reserve(256);
        tag_seq.reserve(256);


        while (getline(is, line)) {
            boost::algorithm::trim(line);
            if (0 == line.size()) continue;
            vector<string> strpair_cont;
            boost::algorithm::split(strpair_cont, line, boost::is_any_of("\t"));
            sent.clear();
            tag_seq.clear();
            for (string &strpair : strpair_cont) {
                string::size_type  delim_pos = strpair.rfind("_");
                assert(delim_pos != string::npos);
                std::string word = strpair.substr(0, delim_pos);
                Index word_id = word_dict.Convert(word);
                Index tag_id = tag_dict.Convert(strpair.substr(delim_pos + 1));
                sent.push_back(word_id);
                tag_seq.push_back(tag_id);
            }
            tmp_samples.emplace_back(sent, tag_seq); // using `pair` construction pair(first_type , second_type)
            ++line_cnt;
            if (0 == line_cnt % 4) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << "lines"; break; }
        }
        swap(tmp_samples, samples);
    }

    void read_training_data_and_build_dicts(istream &is, vector<InstancePair> *samples)
    {
        assert(!word_dict.is_frozen() && !tag_dict.is_frozen());
        BOOST_LOG_TRIVIAL(info) << "reading training data .";
        do_read_dataset(is, *samples);
        // Actually , we should set `TAG_OUTPUT_DIM` and `WORD_DICT_SIZE` at here , but unfortunately ,
        // an special key `UNK` has not been add to the word_dict , and we have to add it after the dict is frozen .
        // What's more , we want frozen the dict only before reading the dev data or testing data .
        // The `Dict` can not set frozen state ! we'd better fix it in the future .
    }

    void read_devel_data(istream &is, vector<InstancePair> *samples)
    {
        if (!word_dict.is_frozen() && !tag_dict.is_frozen()) freeze_dict_and_set_unk();
        BOOST_LOG_TRIVIAL(info) << "reading developing data .";
        do_read_dataset(is, *samples);
    }

    void read_test_data(istream &is, vector<vector<string>> *raw_test_sents, vector<IndexSeq> *test_sents)
    {
        // Read Test data , raw data is also been stored for outputing (because we may get an UNK word and can't can't convert it to origin text )
        // Test data format :
        // Each line is combined with words and delimeters , delimeters can be TAB or SPACE 
        // - Attation : Empty line is also reserved .
        if (!word_dict.is_frozen() && !tag_dict.is_frozen()) freeze_dict_and_set_unk();
        BOOST_LOG_TRIVIAL(info) << "reading test data .";
        vector<IndexSeq> tmp_sents;
        vector<vector<string>> tmp_raw_sents;
        tmp_sents.reserve(0xFFFF); // 60k sents 
        tmp_raw_sents.reserve(0xFFFF);
        string line;
        while (getline(is, line))
        {
            vector<string> words_seq;
            boost::split(words_seq, line, boost::is_any_of("\t "));
            tmp_raw_sents.push_back(words_seq);
            unsigned seq_len = words_seq.size();
            tmp_sents.emplace_back(seq_len); // using constructor `vector(nr_num)` => push_back(vector<int>(nr_words)) 
            IndexSeq &words_index_seq = tmp_sents.back();
            for (unsigned i = 0; i < seq_len; ++i) words_index_seq[i] = word_dict.Convert(words_seq[i]);
        }
        swap(*test_sents, tmp_sents);
        swap(*raw_test_sents, tmp_raw_sents);
    }

    /*************************MODEL HANDLER***********************************/

    /* Function : freeze_dict_and_set_unk_and_model_parameters
    ** This should be call before reading developing data or test data (just once at first time ).
    **/
    void freeze_dict_and_set_unk()
    {
        if (word_dict.is_frozen() && tag_dict.is_frozen()) return; // has been frozen
        tag_dict.Freeze();
        word_dict.Freeze();
        // set unk
        word_dict.SetUnk(UNK_STR);
        // tag dict do not set unk . if has unkown tag , we think it is the dataset error and  should be fixed .
        tag_dict.SetUnk(UNK_STR); // JUST to TEST the program logic !
        UNK = word_dict.Convert(UNK_STR); // get unk id at model (may be usefull for debugging)
    }

    void finish_read_training_data() { freeze_dict_and_set_unk(); }

    void print_model_info()
    {
        cout << "------------Model structure info-----------\n"
            << "vocabulary(word dict) size : " << WORD_DICT_SIZE << " with dimension : " << INPUT_DIM << "\n"
            << "LSTM hidden layer dimension : " << LSTM_HIDDEN_DIM << " , has " << LSTM_LAYER << " layers\n"
            << "TAG hidden layer dimension : " << TAG_HIDDEN_DIM << "\n"
            << "output dimention(tags number) : " << TAG_OUTPUT_DIM << "\n"
            << "--------------------------------------------" << endl;
    }

    void build_model_structure(const po::variables_map& conf, bool is_print_model_info = true)
    {
        LSTM_LAYER = conf["lstm_layers"].as<unsigned>();
        INPUT_DIM = conf["input_dim"].as<unsigned>();
        LSTM_HIDDEN_DIM = conf["lstm_hidden_dim"].as<unsigned>();
        TAG_HIDDEN_DIM = conf["tag_dim"].as<unsigned>();
        // TAG_OUTPUT_DIM and WORD_DICT_SIZE is according to the dict size .
        TAG_OUTPUT_DIM = tag_dict.size();
        WORD_DICT_SIZE = word_dict.size();
        if (0 == TAG_OUTPUT_DIM || !tag_dict.is_frozen() || !word_dict.is_frozen()) {
            BOOST_LOG_TRIVIAL(error) << "`finish_read_training_data` should be call before build model structure \n Exit!";
            abort();
        }

        m = new Model();
        l2r_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);
        r2l_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);

        words_lookup_param = m->add_lookup_parameters(WORD_DICT_SIZE, { INPUT_DIM });
        l2r_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
        r2l_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
        tag_hidden_b_param = m->add_parameters({ TAG_HIDDEN_DIM });

        tag_output_w_param = m->add_parameters({ TAG_OUTPUT_DIM , TAG_HIDDEN_DIM });
        tag_output_b_param = m->add_parameters({ TAG_OUTPUT_DIM });

        if (is_print_model_info) print_model_info();
    }


    void load_wordembedding_from_word2vec_txt_format(const string &fpath)
    {
        // TODO set lookup parameters from outer word embedding
        // using words_loopup_param.Initialize( word_id , value_vector )
    }

    void save_model(ostream &os)
    {
        // This saving order is important !
        // 1. model structure parameters : WORD_DICT_SIZE , INPUT_DIM , LSTM_LAYER , LSTM_HIDDEN_DIM , TAG_HIDDEN_DIM , TAG_OUTPUT_DIM
        // 2. Dict : word_dict , tag_dict 
        // 3. Model of cnn
        boost::archive::text_oarchive to(os);
        to << WORD_DICT_SIZE << INPUT_DIM
            << LSTM_LAYER << LSTM_HIDDEN_DIM
            << TAG_HIDDEN_DIM << TAG_OUTPUT_DIM;

        to << word_dict << tag_dict;

        to << *m;
        BOOST_LOG_TRIVIAL(info) << "saving model done .";
    }

    void load_model(istream &is)
    {
        // Firstly ,  we should load data as the saving data order .
        // What's more , before load `model` , we should build the model structure as same as which has been saved .
        boost::archive::text_iarchive ti(is);
        // 1. load structure data and dict 
        ti >> WORD_DICT_SIZE >> INPUT_DIM
            >> LSTM_LAYER >> LSTM_HIDDEN_DIM
            >> TAG_HIDDEN_DIM >> TAG_OUTPUT_DIM;

        ti >> word_dict >> tag_dict;

        assert(WORD_DICT_SIZE == word_dict.size() && TAG_OUTPUT_DIM == tag_dict.size());

        // 2. build model structure 
        po::variables_map var;
        var.insert({ make_pair(string("lstm_layers") , po::variable_value(boost::any(LSTM_LAYER) , false))  ,
          make_pair(string("input_dim") , po::variable_value(boost::any(INPUT_DIM) , false)) ,
          make_pair(string("lstm_hidden_dim") , po::variable_value(boost::any(LSTM_HIDDEN_DIM) ,false)) ,
          make_pair(string("tag_dim") , po::variable_value(boost::any(TAG_HIDDEN_DIM) , false))
        });
        build_model_structure(var);

        // 3. load model parameter
        ti >> *m; // It is strange ! when I try to deserialize the pointer , it crashed ! using Entity avoid it !
              // It should be attention !

        BOOST_LOG_TRIVIAL(info) << "load model done .";
    }

    /****************************Train , Devel , Predict*********************************/

    Expression negative_loglikelihood(const IndexSeq *p_sent, const IndexSeq *p_tag_seq, ComputationGraph *p_cg, Stat *p_stat = nullptr)
    {
        const unsigned sent_len = p_sent->size();
        ComputationGraph &cg = *p_cg;
        // New graph , ready for new sentence 
        l2r_builder->new_graph(cg);
        l2r_builder->start_new_sequence();
        r2l_builder->new_graph(cg);
        r2l_builder->start_new_sequence();

        // Add parameters to cg
        Expression l2r_tag_hidden_w_exp = parameter(cg, l2r_tag_hidden_w_param);
        Expression r2l_tag_hidden_w_exp = parameter(cg, r2l_tag_hidden_w_param);
        Expression tag_hidden_b_exp = parameter(cg, tag_hidden_b_param);

        Expression tag_output_w_exp = parameter(cg, tag_output_w_param);
        Expression tag_output_b_exp = parameter(cg, tag_output_b_param);

        // Some container
        vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
        vector<Expression> word_lookup_exp_cont(sent_len); // for storing word lookup(embedding) expression for every word(index) in sentence
        vector<Expression> l2r_lstm_output_exp_cont(sent_len); // for storing left to right lstm output(deepest hidden layer) expression for every timestep
        vector<Expression> r2l_lstm_output_exp_cont(sent_len); // right to left 

        // build computation graph(also get output expression) for left to right LSTM
        // 1. get word embeddings for sent 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
            word_lookup_exp_cont[i] = noise(word_lookup_exp, 0.1f);
        }
        Expression SOS_EXP = lookup(cg, words_lookup_param, SOS);
        Expression EOS_EXP = lookup(cg, words_lookup_param, EOS);

        // 2. left 2 right , calc Expression of every timestep of LSTM
        l2r_builder->add_input(SOS_EXP);
        for (unsigned i = 0; i < sent_len; ++i) {
            l2r_lstm_output_exp_cont[i] = l2r_builder->add_input(word_lookup_exp_cont[i]);
        }

        // 3. right 2 left , calc Expression of every timestep of LSTM
        r2l_builder->add_input(EOS_EXP);
        for (int i = static_cast<int>(sent_len) - 1; i >= 0; --i) {
            // should be int , or never stop
            r2l_lstm_output_exp_cont[i] = r2l_builder->add_input(word_lookup_exp_cont[i]);
        }

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            // rectify is suggested as activation function
            Expression tag_hidden_layer_output_at_timestep_t = cnn::expr::rectify(affine_transform({ tag_hidden_b_exp,
              l2r_tag_hidden_w_exp, l2r_lstm_output_exp_cont[i],
              r2l_tag_hidden_w_exp, r2l_lstm_output_exp_cont[i] }));
            Expression tag_output_layer_output_at_timestep_t = affine_transform({ tag_output_b_exp ,
              tag_output_w_exp , tag_hidden_layer_output_at_timestep_t });

            // if statistic , calc output at timestep t
            if (p_stat != nullptr)
            {
                vector<float> output_values = as_vector(cg.incremental_forward());
                float max_value = output_values[0];
                unsigned tag_id_with_max_value = 0;
                for (unsigned i = 1; i < TAG_OUTPUT_DIM; ++i)
                {
                    if (max_value < output_values[i])
                    {
                        max_value = output_values[i];
                        tag_id_with_max_value = i;
                    }
                }
                ++(p_stat->total_tags); // == ++stat->total_tags ;
                if (tag_id_with_max_value == p_tag_seq->at(i)) ++(p_stat->correct_tags);
            }
            err_exp_cont[i] = pickneglogsoftmax(tag_output_layer_output_at_timestep_t, p_tag_seq->at(i));
        }

        // build the finally loss 
        return sum(err_exp_cont); // in fact , no need to return . just to avoid a warning .
    }

    void do_predict(const IndexSeq *p_sent, ComputationGraph *p_cg, IndexSeq *p_predict_tag_seq)
    {
        // The main structure is just a copy from build_bilstm4tagging_graph2train! 
        const unsigned sent_len = p_sent->size();
        ComputationGraph &cg = *p_cg;
        // New graph , ready for new sentence 
        l2r_builder->new_graph(cg);
        l2r_builder->start_new_sequence();
        r2l_builder->new_graph(cg);
        r2l_builder->start_new_sequence();

        // Add parameters to cg
        Expression l2r_tag_hidden_w_exp = parameter(cg, l2r_tag_hidden_w_param);
        Expression r2l_tag_hidden_w_exp = parameter(cg, r2l_tag_hidden_w_param);
        Expression tag_hidden_b_exp = parameter(cg, tag_hidden_b_param);

        Expression tag_output_w_exp = parameter(cg, tag_output_w_param);
        Expression tag_output_b_exp = parameter(cg, tag_output_b_param);

        // Some container
        vector<Expression> word_lookup_exp_cont(sent_len); // for storing word lookup(embedding) expression for every word(index) in sentence
        vector<Expression> l2r_lstm_output_exp_cont(sent_len); // for storing left to right lstm output(deepest hidden layer) expression for every timestep
        vector<Expression> r2l_lstm_output_exp_cont(sent_len); // right to left 

        // build computation graph(also get output expression) for left to right LSTM
        // 1. get word embeddings for sent 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            word_lookup_exp_cont[i] = lookup(cg, words_lookup_param, p_sent->at(i));
        }
        Expression SOS_EXP = lookup(cg, words_lookup_param, SOS);
        Expression EOS_EXP = lookup(cg, words_lookup_param, EOS);

        // 2. left 2 right , calc Expression of every timestep of LSTM
        l2r_builder->add_input(SOS_EXP);
        for (unsigned i = 0; i < sent_len; ++i)
            l2r_lstm_output_exp_cont[i] = l2r_builder->add_input(word_lookup_exp_cont[i]);

        // 3. right 2 left , calc Expression of every timestep of LSTM
        r2l_builder->add_input(EOS_EXP);
        for (int i = static_cast<int>(sent_len) - 1; i >= 0; --i) // should be int , or never stop
            r2l_lstm_output_exp_cont[i] = r2l_builder->add_input(word_lookup_exp_cont[i]);

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression tag_hidden_layer_output_at_timestep_t = tanh(affine_transform({ tag_hidden_b_exp,
              l2r_tag_hidden_w_exp, l2r_lstm_output_exp_cont[i],
              r2l_tag_hidden_w_exp, r2l_lstm_output_exp_cont[i] }));
            Expression tag_output_layer_output_at_timestep_t = affine_transform({ tag_output_b_exp ,
              tag_output_w_exp , tag_hidden_layer_output_at_timestep_t });

            vector<float> output_values = as_vector(cg.incremental_forward());
            float max_value = output_values[0];
            unsigned tag_id_with_max_value = 0;
            for (unsigned i = 1; i < TAG_OUTPUT_DIM; ++i)
            {
                if (max_value < output_values[i])
                {
                    max_value = output_values[i];
                    tag_id_with_max_value = i;
                }
            }
            p_predict_tag_seq->push_back(tag_id_with_max_value);
        }
    }

    void train(const vector<InstancePair> *p_samples, unsigned max_epoch, const vector<InstancePair> *p_dev_samples = nullptr,
        const unsigned do_devel_freq = 3)
    {
        unsigned nr_samples = p_samples->size();

        BOOST_LOG_TRIVIAL(info) << "Train at " << nr_samples << " instances .\n";

        vector<unsigned> access_order(nr_samples);
        for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

        SimpleSGDTrainer sgd = SimpleSGDTrainer(m);

        double total_time_cost_in_seconds = 0.;
        for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
        {
            // shuffle samples by random access order
            shuffle(access_order.begin(), access_order.end(), *rndeng);

            // For loss , accuracy , time cost report
            Stat training_stat_per_report, training_stat_per_epoch;
            unsigned report_freq = 10000;

            // training for an epoch
            training_stat_per_report.start_time_stat();
            training_stat_per_epoch.time_start = training_stat_per_epoch.time_start;
            for (unsigned i = 0; i < nr_samples; ++i)
            {
                const InstancePair &instance_pair = p_samples->at(access_order[i]);
                // using negative_loglikelihood loss to build model
                const IndexSeq *p_sent = &instance_pair.first,
                    *p_tag_seq = &instance_pair.second;
                ComputationGraph cg;
                negative_loglikelihood(p_sent, p_tag_seq, &cg, &training_stat_per_report);
                training_stat_per_report.loss += as_scalar(cg.forward());
                cg.backward();
                sgd.update(1.0);

                if (0 == (i + 1) % report_freq) // Report 
                {
                    training_stat_per_report.end_time_stat();
                    BOOST_LOG_TRIVIAL(info) << i + 1 << " instances have been trained , with E = "
                        << training_stat_per_report.get_E()
                        << " , ACC = " << training_stat_per_report.get_acc() * 100
                        << " % with time cost " << training_stat_per_report.get_time_cost_in_seconds()
                        << " s .";
                    training_stat_per_epoch += training_stat_per_report;
                    training_stat_per_report.clear();
                    training_stat_per_report.start_time_stat();
                }
            }

            // End of an epoch 
            //sgd.status();
            sgd.update_epoch();

            training_stat_per_epoch.end_time_stat();
            training_stat_per_epoch += training_stat_per_report;

            BOOST_LOG_TRIVIAL(info) << "Epoch " << nr_epoch + 1 << " finished . "
                << nr_samples << " instances has been trained ."
                << " For this epoch , E = "
                << training_stat_per_epoch.get_E() << " , ACC = " << training_stat_per_epoch.get_acc() * 100
                << " % with total time cost " << training_stat_per_epoch.get_time_cost_in_seconds()
                << " s ."
                << " total tags : " << training_stat_per_epoch.total_tags
                << " correct tags : " << training_stat_per_epoch.correct_tags;
            total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();

            // If developing samples is available , do `devel` to get model training effect . 
            if (p_dev_samples != nullptr && 0 == (nr_epoch + 1) % do_devel_freq) devel(p_dev_samples);
        }
        BOOST_LOG_TRIVIAL(info) << "Training finished with cost " << total_time_cost_in_seconds << " s .";
    }

    double devel(const vector<InstancePair> *dev_samples)
    {
        unsigned nr_samples = dev_samples->size();
        BOOST_LOG_TRIVIAL(info) << "Validation at " << nr_samples << " instances .\n";
        Stat acc_stat;
        acc_stat.start_time_stat();
        for (const InstancePair &instance_pair : *dev_samples)
        {
            ComputationGraph cg;
            IndexSeq predict_tag_seq;
            const IndexSeq &sent = instance_pair.first,
                &tag_seq = instance_pair.second;
            do_predict(&sent, &cg, &predict_tag_seq);
            assert(predict_tag_seq.size() == tag_seq.size());
            for (unsigned i = 0; i < tag_seq.size(); ++i)
            {
                ++acc_stat.total_tags;
                if (tag_seq[i] == predict_tag_seq[i]) ++acc_stat.correct_tags;
            }
        }
        acc_stat.end_time_stat();
        BOOST_LOG_TRIVIAL(info) << "Validation finished . ACC = "
            << acc_stat.get_acc() * 100 << " % "
            << ", with time cosing " << acc_stat.get_time_cost_in_seconds() << " s .";
        return acc_stat.get_acc();
    }

    void predict(istream &is, ostream &os)
    {
        const string SPLIT_DELIMITER = "\t";
        vector<vector<string>> raw_instances;
        vector<IndexSeq> index_instances;
        read_test_data(is, &raw_instances, &index_instances);
        assert(raw_instances.size() == index_instances.size());
        for (unsigned int i = 0; i < raw_instances.size(); ++i)
        {
            vector<string> *p_raw_sent = &raw_instances.at(i);
            if (0 == p_raw_sent->size())
            {
                os << "\n";
                continue;
            }
            IndexSeq *p_sent = &index_instances.at(i);
            IndexSeq predict_seq;
            ComputationGraph cg;
            do_predict(p_sent, &cg, &predict_seq);
            // output the result directly
            os << p_raw_sent->at(0) << "_" << tag_dict.Convert(predict_seq.at(0));
            for (unsigned k = 1; k < p_raw_sent->size(); ++k)
            {
                os << SPLIT_DELIMITER
                    << p_raw_sent->at(k) << "_" << tag_dict.Convert(predict_seq.at(k));
            }
            os << "\n";
        }
    }
};

void init_command_line_options(int argc, char* argv[], po::variables_map* conf) {
    po::options_description opts("Configuration");
    opts.add_options()
        ("training_data", po::value<std::string>(), "The path to the training data.")
        ("devel_data", po::value<std::string>(), "The path to the development data.")
        ("test_data", po::value<std::string>(), "The path to the test data.")
        ("train", "use to specify to perform training process.")
        ("devel_freq", po::value<unsigned>()->default_value(2), "The frequence for testing model on developing data during traing")
        ("model", po::value<std::string>(), "use to specify the model name.")
        ("max_epoch", po::value<unsigned>()->default_value(4), "The epoch number for training")
        ("input_dim", po::value<unsigned>()->default_value(50), "The dimension for input word embedding.")
        ("lstm_layers", po::value<unsigned>()->default_value(1), "The number of layers in bi-LSTM.")
        ("lstm_hidden_dim", po::value<unsigned>()->default_value(100), "The dimension for LSTM output.")
        ("tag_dim", po::value<unsigned>()->default_value(32), "The dimension for tag.")
        ("help,h", "Show help information.")
        ;

    po::store(po::parse_command_line(argc, argv, opts), *conf);
    if (conf->count("help")) {
        std::cerr << opts << std::endl;
        exit(1);
    }

    if (conf->count("training_data") == 0) {
        BOOST_LOG_TRIVIAL(error) << "Please specify --training_data : "
            << "this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.";
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    // argv :
    // --cnn-mem 128 --training_data C:/data/ltp-data/ner/pku-weibo-train.pos --devel_data C:/data/ltp-data/ner/pku-weibo-holdout.pos --test_data test.pos
    cnn::Initialize(argc, argv, 1234); // MUST , or no memory is allocated ! Also the the random seed.
    
    po::variables_map conf;
    init_command_line_options(argc, argv, &conf);

    // -
    // BUILD MODEL
    // -
    // declare model 
    BILSTMModel4Tagging tagging_model;

    // Reading training data , build word dict and tag dict 
    ifstream train_is(conf["training_data"].as<std::string>());
    if (!train_is) {
        BOOST_LOG_TRIVIAL(error) << "failed to open training: " << conf["training_data"].as<std::string>();
        return -1;
    }
    vector<InstancePair> training_samples;
    cnn::Dict &word_dict = tagging_model.word_dict,
        &tag_dict = tagging_model.tag_dict;
    tagging_model.read_training_data_and_build_dicts(train_is, &training_samples);
    train_is.close();

    // After reading training data , call `finish_read_training_data`
    tagging_model.finish_read_training_data();

    // Build model structure 
    tagging_model.build_model_structure(conf);

    // Reading developing data .
    std::vector<InstancePair> devel_samples, *p_devel_samples;
    if (0 != conf.count("devel_data"))
    {
        std::ifstream devel_is(conf["devel_data"].as<std::string>());
        if (!devel_is) {
            BOOST_LOG_TRIVIAL(error) << "failed to open devel file: " << conf["devel_data"].as<std::string>();
            return -1;
        }
        tagging_model.read_devel_data(devel_is, &devel_samples);
        devel_is.close();
        p_devel_samples = &devel_samples;
    }
    else p_devel_samples = nullptr;
    unsigned devel_freq = conf["devel_freq"].as<unsigned>();

    // Train 
    unsigned epoch = conf["max_epoch"].as<unsigned>();
    tagging_model.train(&training_samples, epoch, p_devel_samples, devel_freq);

    // Test 
    // - Here we use no-tag (has been segmented) text(utf-8 encoded) as the test file .
    // - such as : 我 是 中国 人 。
    // For the dataset with tag , just using `devel` !
    string test_data_path = conf["test_data"].as<string>();
    ifstream test_is(test_data_path);
    if (!test_is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open test data at '" << test_data_path << "' \n Exit .";
        return -1;
    }
    tagging_model.predict(test_is, cout);
    test_is.close();

    // Save model
    string model_path;
    if (0 == conf.count("model"))
    {
        ostringstream oss;
        oss << "tagging_" << tagging_model.INPUT_DIM << "_" << tagging_model.LSTM_HIDDEN_DIM
            << "_" << tagging_model.TAG_HIDDEN_DIM << ".model";
        model_path = oss.str();
    }
    else model_path = conf["model"].as<string>();
    ofstream os(model_path);
    if (!os)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open model path at '" << model_path << "'. \n Exit .";
        return -1;
    }
    tagging_model.save_model(os);
    os.close();

    // Load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n Exit .";
        return -1;
    }
    BILSTMModel4Tagging another_model;
    another_model.load_model(is);
    another_model.devel(p_devel_samples); // Get the same result , it is OK .
    is.close();
    getchar(); // pause ;
    return 0;
}
