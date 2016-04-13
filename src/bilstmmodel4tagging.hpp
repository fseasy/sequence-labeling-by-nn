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
#include <functional>
#include <algorithm>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options.hpp>

#include "utf8processing.hpp"
#include "dict_wrapper.hpp"
#include "stat4postagger.hpp"

using namespace std;
using namespace cnn;
namespace po = boost::program_options;


/********************************
 *BILSTMModel4Tagging (for postagging currently)
 *tagging model based on CNN library , do as CNN examples `tag-bilstm.cc`
 *
 * ******************************/
namespace slnn{
struct BILSTMModel4Tagging
{
    // model 
    Model *m;
    LSTMBuilder* l2r_builder;
    LSTMBuilder* r2l_builder;

    // paramerters
    LookupParameters *words_lookup_param;
    LookupParameters *tags_lookup_param;
    Parameters *l2r_tag_hidden_w_param;
    Parameters *r2l_tag_hidden_w_param;
    Parameters *pretag_tag_hidden_w_param;
    Parameters *tag_hidden_b_param;

    Parameters *tag_output_w_param;
    Parameters *tag_output_b_param;

    // model structure : using const !(in-class initilization)

    unsigned INPUT_DIM; // word embedding dimension
    unsigned TAG_EMBEDDING_DIM;
    unsigned LSTM_LAYER;
    unsigned LSTM_HIDDEN_DIM;
    unsigned TAG_HIDDEN_DIM;
    //--- need to be counted from training data or loaded from model .
    unsigned TAG_DICT_SIZE; 
    unsigned TAG_OUTPUT_DIM; // it my be not equal to TAG_DICT_SIZE
    unsigned WORD_DICT_SIZE;

    // model saving
    float best_acc;
    stringstream best_model_tmp_ss;

    // others 
    cnn::Dict word_dict;
    cnn::Dict tag_dict;
    DictWrapper word_dict_wrapper;
    const string SOS_STR = "<START_OF_SEQUENCE_REPR>";
    const string EOS_STR = "<END_OF_SEQUENCE_REPR>";
    const string SOS_TAG_STR = "<STAET_OF_TAG_SEQUENCE_REPR>";
    const string UNK_STR = "<UNK_REPR>"; // should add a unknown token into the lexicon. 
    Index SOS;
    Index EOS;
    Index SOS_TAG;
    Index UNK;

    static const string number_transform_str;
    static const size_t length_transform_str;

    BILSTMModel4Tagging() :
        m(nullptr), l2r_builder(nullptr), r2l_builder(nullptr) , best_acc(0.), best_model_tmp_ss() ,
        word_dict_wrapper(word_dict)
    {}

    ~BILSTMModel4Tagging()
    {
        if (m) delete m;
        if (l2r_builder) delete l2r_builder;
        if (r2l_builder) delete r2l_builder;
    }

    /*************************READING DATA **********************************/

    string replace_number(const string &str)
    {
        string tmp_str = str;
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
                // Parse Number to specific string
                word = replace_number(word);
                Index word_id = word_dict_wrapper.Convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
                Index tag_id = tag_dict.Convert(strpair.substr(delim_pos + 1));
                sent.push_back(word_id);
                tag_seq.push_back(tag_id);
            }
            tmp_samples.emplace_back(sent, tag_seq); // using `pair` construction pair(first_type , second_type)
            ++line_cnt;
            if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << "lines";  }
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
        if (!word_dict.is_frozen() && !tag_dict.is_frozen()) add_special_flag_and_freeze_dict();
        BOOST_LOG_TRIVIAL(info) << "reading developing data .";
        do_read_dataset(is, *samples);
    }

    void read_test_data(istream &is, vector<vector<string>> *raw_test_sents, vector<IndexSeq> *test_sents)
    {
        // Read Test data , raw data is also been stored for outputing (because we may get an UNK word and can't can't convert it to origin text )
        // Test data format :
        // Each line is combined with words and delimeters , delimeters can be TAB or SPACE 
        // - Attation : Empty line is also reserved .
        if (!word_dict.is_frozen() && !tag_dict.is_frozen()) add_special_flag_and_freeze_dict();
        BOOST_LOG_TRIVIAL(info) << "reading test data .";
        vector<IndexSeq> tmp_sents;
        vector<vector<string>> tmp_raw_sents;
        tmp_sents.reserve(0xFFFF); // 60k sents 
        tmp_raw_sents.reserve(0xFFFF);
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
                words_index_seq[i] = word_dict.Convert(number_transed_word);
            }
                
        }
        swap(*test_sents, tmp_sents);
        swap(*raw_test_sents, tmp_raw_sents);
    }

    /*************************MODEL HANDLER***********************************/

    /* Function : add_special_flag_and_freeze_dict_and_model_parameters
    ** This should be call before reading developing data or test data (just once at first time ).
    **/
    void add_special_flag_and_freeze_dict()
    {
        if (word_dict.is_frozen() && tag_dict.is_frozen()) return; // has been frozen
        // First , add flag for start , end of word sequence and start of tag sequence  
        SOS = word_dict_wrapper.Convert(SOS_STR);
        EOS = word_dict_wrapper.Convert(EOS_STR);
        SOS_TAG = tag_dict.Convert(SOS_TAG_STR); // The SOS_TAG should just using in tag hidden layer , and never in output ! 
        // Freeze
        tag_dict.Freeze();
        word_dict_wrapper.Freeze();
        // set unk
        word_dict_wrapper.SetUnk(UNK_STR);
        // tag dict do not set unk . if has unkown tag , we think it is the dataset error and  should be fixed .
        UNK = word_dict_wrapper.Convert(UNK_STR); // get unk id at model (may be usefull for debugging)
    }

    void finish_read_training_data() { add_special_flag_and_freeze_dict(); }

    void print_model_info()
    {
        cout << "------------Model structure info-----------\n"
            << "vocabulary(word dict) size : " << WORD_DICT_SIZE << " with dimension : " << INPUT_DIM << "\n"
            << "tag lookup dimension : " << TAG_EMBEDDING_DIM << "\n"
            << "LSTM hidden layer dimension : " << LSTM_HIDDEN_DIM << " , has " << LSTM_LAYER << " layers\n"
            << "TAG hidden layer dimension : " << TAG_HIDDEN_DIM << "\n"
            << "output dimention(tags number) : " << TAG_OUTPUT_DIM << "\n"
            << "--------------------------------------------" << endl;
    }

    void build_model_structure(const po::variables_map& conf, bool is_print_model_info = true)
    {
        LSTM_LAYER = conf["lstm_layers"].as<unsigned>();
        INPUT_DIM = conf["input_dim"].as<unsigned>();
        TAG_EMBEDDING_DIM = conf["tag_embedding_dim"].as<unsigned>();
        LSTM_HIDDEN_DIM = conf["lstm_hidden_dim"].as<unsigned>();
        TAG_HIDDEN_DIM = conf["tag_dim"].as<unsigned>();
        // TAG_OUTPUT_DIM and WORD_DICT_SIZE is according to the dict size .
        TAG_DICT_SIZE = tag_dict.size();
        TAG_OUTPUT_DIM = TAG_DICT_SIZE - 1 ;
        WORD_DICT_SIZE = word_dict.size();
        if (0 == TAG_OUTPUT_DIM || !tag_dict.is_frozen() || !word_dict.is_frozen()) {
            BOOST_LOG_TRIVIAL(error) << "`finish_read_training_data` should be call before build model structure \n Exit!";
            abort();
        }

        m = new Model();
        l2r_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);
        r2l_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);

        words_lookup_param = m->add_lookup_parameters(WORD_DICT_SIZE, { INPUT_DIM }); // ADD for PRE_TAG
        tags_lookup_param = m->add_lookup_parameters(TAG_DICT_SIZE, { TAG_EMBEDDING_DIM }); // having `SOS_TAG`
        l2r_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
        r2l_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
        pretag_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , TAG_EMBEDDING_DIM }); // ADD for PRE_TAG
        tag_hidden_b_param = m->add_parameters({ TAG_HIDDEN_DIM });

        tag_output_w_param = m->add_parameters({ TAG_OUTPUT_DIM , TAG_HIDDEN_DIM }); // no `SOS_TAG`
        tag_output_b_param = m->add_parameters({ TAG_OUTPUT_DIM });

        if (is_print_model_info) print_model_info();
    }


    void load_wordembedding_from_word2vec_txt_format(istream &is)
    {
        // set lookup parameters from outer word embedding
        // using words_loopup_param.Initialize( word_id , value_vector )
        string line;
        vector<string> split_cont;
        split_cont.reserve(INPUT_DIM + 1); // word + numbers 
        getline(is, line); // first line is the infomation !
        boost::split(split_cont, line, boost::is_any_of(" "));
        assert(2 == split_cont.size() && stoul(split_cont.at(1)) == INPUT_DIM);
        unsigned long long line_cnt = 1 ;
        unsigned long long words_cnt_hit = 0;
        vector<float> embedding_vec(INPUT_DIM , 0.f);
        while (getline(is, line))
        {
            ++line_cnt;
            boost::trim_right(line);
            boost::split(split_cont, line , boost::is_any_of(" "));
            if (INPUT_DIM + 1 != split_cont.size())
            {
                BOOST_LOG_TRIVIAL(info) << "bad word dimension : `" << split_cont.size() - 1 << "` at line " << line_cnt;
                continue;
            }
            string &word = split_cont.at(0);
            Index word_id = word_dict.Convert(word);
            if (word_id != UNK)
            {
                ++words_cnt_hit;
                transform(split_cont.cbegin() + 1, split_cont.cend(), embedding_vec.begin(),
                    [](const string &f_str) { return stof(f_str);});
                words_lookup_param->Initialize(word_id, embedding_vec);
            }
        }
        BOOST_LOG_TRIVIAL(info) << "Initialize word embedding done . " << words_cnt_hit << "/" << WORD_DICT_SIZE
            << " words emebdding has been loading .";
    }

    void save_model(ostream &os)
    {
        // This saving order is important !
        // 1. model structure parameters : WORD_DICT_SIZE , INPUT_DIM , LSTM_LAYER , LSTM_HIDDEN_DIM , TAG_HIDDEN_DIM , TAG_OUTPUT_DIM
        //                                 TAG_EMBEDDING_DIM , TAG_DICT_SIZE
        // 2. Dict : word_dict , tag_dict 
        // 3. Model of cnn
        boost::archive::text_oarchive to(os);
        to << WORD_DICT_SIZE << INPUT_DIM
            << LSTM_LAYER << LSTM_HIDDEN_DIM
            << TAG_HIDDEN_DIM << TAG_OUTPUT_DIM
            << TAG_EMBEDDING_DIM << TAG_DICT_SIZE ; // ADD for PRE_TAG

        to << word_dict << tag_dict;
        if (0 != best_model_tmp_ss.rdbuf()->in_avail())
        {
            boost::archive::text_iarchive ti(best_model_tmp_ss);
            ti >> *m;
            ; // if best model is not save to the temporary stringstream , we should firstly save it !
        }
        
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
            >> TAG_HIDDEN_DIM >> TAG_OUTPUT_DIM 
            >> TAG_EMBEDDING_DIM >> TAG_DICT_SIZE ; // ADD for PRE_TAG

        ti >> word_dict >> tag_dict;

        assert(WORD_DICT_SIZE == word_dict.size() && TAG_DICT_SIZE == tag_dict.size());
       
        // SET VALUE for special flag
        SOS = word_dict.Convert(SOS_STR);
        EOS = word_dict.Convert(EOS_STR);
        SOS_TAG = tag_dict.Convert(SOS_TAG_STR); // The SOS_TAG should just using in tag hidden layer , and never in output ! 
        UNK = word_dict.Convert(UNK_STR); // get unk id at model (may be usefull for debugging)

        // 2. build model structure 
        po::variables_map var;
        var.insert({ make_pair(string("lstm_layers") , po::variable_value(boost::any(LSTM_LAYER) , false))  ,
          make_pair(string("input_dim") , po::variable_value(boost::any(INPUT_DIM) , false)) ,
          make_pair(string("lstm_hidden_dim") , po::variable_value(boost::any(LSTM_HIDDEN_DIM) ,false)) ,
          make_pair(string("tag_dim") , po::variable_value(boost::any(TAG_HIDDEN_DIM) , false)) ,
          make_pair(string("tag_embedding_dim") , po::variable_value(boost::any(TAG_EMBEDDING_DIM) , false ))
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
        Expression pretag_tag_hidden_w_exp = parameter(cg, pretag_tag_hidden_w_param);
        Expression tag_hidden_b_exp = parameter(cg, tag_hidden_b_param);

        Expression tag_output_w_exp = parameter(cg, tag_output_w_param);
        Expression tag_output_b_exp = parameter(cg, tag_output_b_param);

        // Some container
        vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
        vector<Expression> word_lookup_exp_cont(sent_len); // for storing word lookup(embedding) expression for every word(index) in sentence
        vector<Expression> l2r_lstm_output_exp_cont(sent_len); // for storing left to right lstm output(deepest hidden layer) expression for every timestep
        vector<Expression> r2l_lstm_output_exp_cont(sent_len); // right to left 
        vector<Expression> pretag_lookup_exp_cont(sent_len); // ADD for PRE_TAG

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
        for (unsigned i = 0; i < sent_len; ++i) 
        {
            l2r_lstm_output_exp_cont[i] = l2r_builder->add_input(word_lookup_exp_cont[i]);
        }

        // 3. right 2 left , calc Expression of every timestep of LSTM
        r2l_builder->add_input(EOS_EXP);
        for (int i = static_cast<int>(sent_len) - 1; i >= 0; --i) 
        {
            // should be int , or never stop
            r2l_lstm_output_exp_cont[i] = r2l_builder->add_input(word_lookup_exp_cont[i]);
        }

        // 4. prepare for PRE_TAG embedding
        Expression SOS_TAG_EXP = lookup(cg, tags_lookup_param, SOS_TAG);
        pretag_lookup_exp_cont[0] = SOS_TAG_EXP ;
        for (unsigned i = 1; i < sent_len ; ++i)
        {
            pretag_lookup_exp_cont[i] = lookup(cg, tags_lookup_param, p_tag_seq->at(i - 1));
        }

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            // rectify is suggested as activation function
            Expression tag_hidden_layer_output_at_timestep_t = cnn::expr::rectify(affine_transform({ tag_hidden_b_exp,
              l2r_tag_hidden_w_exp, l2r_lstm_output_exp_cont[i],
              r2l_tag_hidden_w_exp, r2l_lstm_output_exp_cont[i],
              pretag_tag_hidden_w_exp , pretag_lookup_exp_cont[i]})); // ADD for PRE_TAG
            Expression tag_output_layer_output_at_timestep_t = affine_transform({ tag_output_b_exp ,
              tag_output_w_exp , tag_hidden_layer_output_at_timestep_t });

            // if statistic , calc output at timestep t
            if (p_stat != nullptr)
            {
                vector<float> output_values = as_vector(cg.incremental_forward());
                float max_value = output_values[0];
                Index tag_id_with_max_value = 0;
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
        Expression pretag_tag_hidden_w_exp = parameter(cg, pretag_tag_hidden_w_param);
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

        // 4. set previous tag lookup expression

        Expression pretag_lookup_exp = lookup(cg, tags_lookup_param, SOS_TAG);

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression tag_hidden_layer_output_at_timestep_t = cnn::expr::rectify(affine_transform({ tag_hidden_b_exp,
              l2r_tag_hidden_w_exp, l2r_lstm_output_exp_cont[i],
              r2l_tag_hidden_w_exp, r2l_lstm_output_exp_cont[i],
              pretag_tag_hidden_w_exp , pretag_lookup_exp})); // Add for PRE_TAG
            affine_transform({ tag_output_b_exp , tag_output_w_exp , tag_hidden_layer_output_at_timestep_t }); // only add to CG
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
            // set pretag_lookup_exp for next timestep 
            pretag_lookup_exp = lookup(cg, tags_lookup_param, tag_id_with_max_value);
        }
    }

    void train(const vector<InstancePair> *p_samples, unsigned max_epoch, const vector<InstancePair> *p_dev_samples = nullptr,
        const unsigned long do_devel_freq=50000)
    {
        unsigned nr_samples = p_samples->size();

        BOOST_LOG_TRIVIAL(info) << "Train at " << nr_samples << " instances .\n";

        vector<unsigned> access_order(nr_samples);
        for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

        SimpleSGDTrainer sgd = SimpleSGDTrainer(m);
        unsigned long line_cnt_for_devel = 0;
        unsigned long long total_time_cost_in_seconds = 0ULL ;
        for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
        {
            // shuffle samples by random access order
            shuffle(access_order.begin(), access_order.end(), *rndeng);

            // For loss , accuracy , time cost report
            Stat training_stat_per_report, training_stat_per_epoch;
            unsigned report_freq = 10000;

            // training for an epoch
            training_stat_per_report.start_time_stat();
            training_stat_per_epoch.start_time_stat() ;
            for (unsigned i = 0; i < nr_samples; ++i)
            {
                const InstancePair &instance_pair = p_samples->at(access_order[i]);
                // using negative_loglikelihood loss to build model
                IndexSeq *p_sent = const_cast<IndexSeq *>(&instance_pair.first);
                const IndexSeq *p_tag_seq = &instance_pair.second;
                ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                               // so we have to declaring it as pointer and destroy it handly 
                                                               // before develing.
                // transform low-frequent words to UNK according to the probability
                transform(p_sent->begin(), p_sent->end(), p_sent->begin(),
                    [this](Index word_idx)->Index {return this->word_dict_wrapper.ConvertProbability(word_idx); }) ; // TRANS
                negative_loglikelihood(p_sent, p_tag_seq, cg, &training_stat_per_report);
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
                if (p_dev_samples != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
                {
                    float acc = devel(p_dev_samples);
                    if (acc > best_acc)
                    {
                        BOOST_LOG_TRIVIAL(info) << "Better model found . stash it .";
                        best_acc = acc;
                        best_model_tmp_ss.str(""); // first , clear it's content !
                        boost::archive::text_oarchive to(best_model_tmp_ss);
                        to << *m;
                    }
                    line_cnt_for_devel = 0; // avoid overflow
                }
            }

            // End of an epoch 
            //sgd.status();
            sgd.update_epoch();

            training_stat_per_epoch.end_time_stat();
            training_stat_per_epoch += training_stat_per_report;

            // Output
            long long epoch_time_cost = training_stat_per_epoch.get_time_cost_in_seconds();
            BOOST_LOG_TRIVIAL(info) << "-------- Epoch " << nr_epoch + 1 << " finished . ----------\n"
                << nr_samples << " instances has been trained ."
                << " For this epoch , E = "
                << training_stat_per_epoch.get_E() << " , ACC = " << training_stat_per_epoch.get_acc() * 100
                << " % with total time cost " << epoch_time_cost << " s"
                << "( speed " << training_stat_per_epoch.get_speed_as_kilo_tokens_per_sencond() << " k/s tokens)."
                << " total tags : " << training_stat_per_epoch.total_tags
                << " correct tags : " << training_stat_per_epoch.correct_tags
                << "\n";

            total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
        }
        BOOST_LOG_TRIVIAL(info) << "Training finished with cost " << total_time_cost_in_seconds << " s .";
    }

    float devel(const vector<InstancePair> *dev_samples , ostream *p_error_output_os=nullptr)
    {
        unsigned nr_samples = dev_samples->size();
        BOOST_LOG_TRIVIAL(info) << "Validation at " << nr_samples << " instances .\n";
        unsigned long line_cnt4error_output = 0;
        if (p_error_output_os) *p_error_output_os << "line_nr\tword_index\tword_at_dict\tpredict_tag\ttrue_tag\n";
        Stat acc_stat;
        acc_stat.start_time_stat();
        for (const InstancePair &instance_pair : *dev_samples)
        {
            ++line_cnt4error_output;
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
                else if (p_error_output_os)
                {
                    *p_error_output_os << line_cnt4error_output << "\t" << i << "\t" << word_dict.Convert(sent[i])
                        << "\t" << tag_dict.Convert(predict_tag_seq[i]) << "\t" << tag_dict.Convert(tag_seq[i]) << "\n" ;
                }
            }
        }
        acc_stat.end_time_stat();
        BOOST_LOG_TRIVIAL(info) << "Validation finished . ACC = "
            << acc_stat.get_acc() * 100 << " % "
            << ", with time cosing " << acc_stat.get_time_cost_in_seconds() << " s . " 
            << "(speed " << acc_stat.get_speed_as_kilo_tokens_per_sencond() << " k/s tokens) "
            << "total tags : " << acc_stat.total_tags << " correct tags : " << acc_stat.correct_tags ;
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


const string BILSTMModel4Tagging::number_transform_str = "##";
const size_t BILSTMModel4Tagging::length_transform_str = number_transform_str.length();

} // End of namespace slnn 