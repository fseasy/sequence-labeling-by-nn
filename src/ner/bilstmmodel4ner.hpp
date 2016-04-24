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

#include "modelmodule/layers.h"
#include "utils/utf8processing.hpp"
#include "utils/dict_wrapper.hpp"
#include "utils/stat.hpp"

using namespace std;
using namespace cnn;
namespace po = boost::program_options;


/********************************
 *BILSTMModel4Tagging (for postagging currently)
 *tagging model based on CNN library , do as CNN examples `tag-bilstm.cc`
 *
 * ******************************/
namespace slnn{
struct BILSTMModel4NER
{
    // model 
    Model *m;
    Merge2Layer *input_merge_layer;
    BILSTMLayer *bilstm_layer;
    Merge3Layer *bilstm_pretag_merge_layer;
    DenseLayer *output_linear_layer;

    // paramerters
    // - Lookup parameters
    LookupParameters *words_lookup_param;
    LookupParameters *postags_lookup_param;
    LookupParameters *nertags_lookup_param;

    // - parameters
    // - affine transform at input for word embedding and postag embedding

    // - affline transform at LSTM-OUT & PRE-TAG  params / it is also the TAG HIDDEN layer
    Parameters *pretag_SOS_param;
    

    // model structure 
    
    unsigned WORD_DICT_SIZE; // extra `WORD_UNK`
    unsigned WORD_EMBEDDING_DIM;

    unsigned POSTAG_DICT_SIZE; 
    unsigned POSTAG_EMBEDDING_DIM;

    unsigned LSTM_X_DIM;
    unsigned LSTM_H_DIM;
    unsigned LSTM_LAYER;
    
    
    unsigned NER_DICT_SIZE;
    unsigned NER_TAG_EMBEDDING_DIM; // using for previous-tag

    unsigned NER_LAYER_HIDDEN_DIM;
    unsigned NER_LAYER_OUTPUT_DIM; // will be set after trainning
    

    // model saving
    float best_F1;
    stringstream best_model_tmp_ss;

    // others 
    cnn::Dict word_dict;
    cnn::Dict postag_dict;
    cnn::Dict ner_dict;
    DictWrapper word_dict_wrapper;
    //const string SOS_STR = "<START_OF_SEQUENCE_REPR>";
    //const string EOS_STR = "<END_OF_SEQUENCE_REPR>";
    //const string SOS_TAG_STR = "<STAET_OF_TAG_SEQUENCE_REPR>";
    const string UNK_STR = "<UNK_REPR>"; // should add a unknown token into the lexicon. 
    //Index SOS;
    //Index EOS;
    //Index SOS_TAG;
    Index UNK;

    static const string number_transform_str;
    static const size_t length_transform_str;

    BILSTMModel4NER() :
        m(nullptr), input_merge_layer(nullptr) , bilstm_layer(nullptr),
        bilstm_pretag_merge_layer(nullptr) , output_linear_layer(nullptr) ,
        best_F1(0.f), best_model_tmp_ss() ,
        word_dict_wrapper(word_dict)
    {}

    ~BILSTMModel4NER()
    {
        if (m) delete m;
        if (input_merge_layer) delete input_merge_layer;
        if (bilstm_layer) delete bilstm_layer;
        if (bilstm_pretag_merge_layer) delete bilstm_pretag_merge_layer;
        if (output_linear_layer) delete output_linear_layer;
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

    void do_read_dataset(istream &is, vector<IndexSeq> &sents , vector<IndexSeq> &postag_seqs , vector<IndexSeq> &nertag_seqs)
    {
        // read training data or developing data .
        // data format :
        // Each line is combined with `WORD_TAG`s and delimeters , delimeter is only TAB . `WORD_TAG` should be split to `WORD` and `TAG`
        // Attention : empty line will be skipped 
        unsigned line_cnt = 0;
        string line;
        vector<IndexSeq> tmp_sents;
        vector<IndexSeq> tmp_postag_seqs;
        vector<IndexSeq> tmp_nertag_seqs;
        IndexSeq sent,
            postag_seq , 
            nertag_seq;

        tmp_sents.reserve(0x8000); // 2^15 =  32k pairs 
        tmp_postag_seqs.reserve(0x8000); 
        tmp_nertag_seqs.reserve(0x8000); 
        sent.reserve(256);
        postag_seq.reserve(256);
        nertag_seq.reserve(256);

        while (getline(is, line)) {
            boost::algorithm::trim(line);
            if (0 == line.size()) continue;
            vector<string> parts ;
            boost::algorithm::split(parts, line, boost::is_any_of("\t"));
            sent.clear();
            postag_seq.clear();
            nertag_seq.clear();
            for (string &part : parts) {
                string::size_type postag_pos = part.rfind("/");
                string::size_type nertag_pos = part.rfind("#");
                assert(postag_pos != string::npos && nertag_pos != string::npos);
                string word = part.substr(0, postag_pos);
                string postag = part.substr(postag_pos + 1, nertag_pos - postag_pos - 1);
                string nertag = part.substr(nertag_pos + 1);
                // Parse Number to specific string
                word = replace_number(word);
                Index word_id = word_dict_wrapper.Convert(word); // using word_dict_wrapper , if not frozen , will count the word freqency
                Index postag_id = postag_dict.Convert(postag);
                Index nertag_id = ner_dict.Convert(nertag);
                sent.push_back(word_id);
                postag_seq.push_back(postag_id);
                nertag_seq.push_back(nertag_id);
            }
            tmp_sents.push_back(sent);
            tmp_postag_seqs.push_back(postag_seq);
            tmp_nertag_seqs.push_back(nertag_seq);
            ++line_cnt;
            if (0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << "lines";  }
        }
        swap(tmp_sents, sents);
        swap(tmp_postag_seqs, postag_seqs);
        swap(tmp_nertag_seqs, nertag_seqs);
    }

    void read_training_data_and_build_dicts(istream &is, vector<IndexSeq> *p_sent_seqs , vector<IndexSeq> *p_postag_seqs , 
        vector<IndexSeq> *p_nertag_seqs)
    {
        assert(!word_dict.is_frozen() && !postag_dict.is_frozen() && !ner_dict.is_frozen());
        BOOST_LOG_TRIVIAL(info) << "reading training data .";
        do_read_dataset(is, *p_sent_seqs , *p_postag_seqs , *p_nertag_seqs);
        // Actually , we should set `TAG_OUTPUT_DIM` and `WORD_DICT_SIZE` at here , but unfortunately ,
        // an special key `UNK` has not been add to the word_dict , and we have to add it after the dict is frozen .
        // What's more , we want frozen the dict only before reading the dev data or testing data .
        // The `Dict` can not set frozen state ! we'd better fix it in the future .
    }

    void read_devel_data(istream &is, vector<IndexSeq> *p_dev_sents , vector<IndexSeq> *p_dev_postag_seqs , 
        vector<IndexSeq> *p_dev_nertag_seqs)
    {
        if (!word_dict.is_frozen() && !postag_dict.is_frozen() && !ner_dict.is_frozen()) add_special_flag_and_freeze_dict();
        BOOST_LOG_TRIVIAL(info) << "reading developing data .";
        do_read_dataset(is, *p_dev_sents , *p_dev_postag_seqs , *p_dev_nertag_seqs);
    }

    void read_test_data(istream &is, vector<vector<string>> *p_raw_test_sents, vector<IndexSeq> *p_test_sents ,
        vector<IndexSeq> *p_test_postag_seqs)
    {
        // Read Test data , raw data is also been stored for outputing (because we may get an UNK word and can't can't convert it to origin text )
        // Test data format :
        // Each line is combined with WORD_POSTAG and delimeters , delimeters can be TAB or SPACE 
        // - Attation : Empty line is also reserved .
        if (!word_dict.is_frozen() && !postag_dict.is_frozen() && !ner_dict.is_frozen()) add_special_flag_and_freeze_dict();
        BOOST_LOG_TRIVIAL(info) << "reading test data .";
        vector<vector<string>> tmp_raw_sents;
        vector<IndexSeq> tmp_sents;
        vector<IndexSeq> tmp_postag_seqs;
        vector<string> raw_sent;
        IndexSeq sent,
            postag_seq;

        tmp_raw_sents.reserve(0x4FFF);
        tmp_sents.reserve(0x4FFF); // 16k sents 
        tmp_postag_seqs.reserve(0x4FFF);
        raw_sent.reserve(256);
        sent.reserve(256);
        postag_seq.reserve(256);
        
        string line;
        while (getline(is, line))
        {
            boost::trim(line);
            vector<string> parts;
            boost::split(parts , line, boost::is_any_of("\t"));
            
            raw_sent.resize(0);
            sent.resize(0);
            postag_seq.resize(0);

            for (string &part : parts)
            {
                string::size_type delim_pos = part.rfind("_");
                string raw_word = part.substr(0, delim_pos);
                string postag = part.substr(delim_pos + 1);
                raw_sent.push_back(raw_word);
                string word = replace_number(raw_word);
                sent.push_back(word_dict.Convert(word));
                postag_seq.push_back(postag_dict.Convert(postag));
            }
            tmp_raw_sents.push_back(raw_sent);
            tmp_sents.push_back(sent);
            tmp_postag_seqs.push_back(postag_seq);
                
        }
        swap(*p_raw_test_sents, tmp_raw_sents);
        swap(*p_test_sents, tmp_sents);
        swap(*p_test_postag_seqs, tmp_postag_seqs);
    }

    /*************************MODEL HANDLER***********************************/

    /* Function : add_special_flag_and_freeze_dict_and_model_parameters
    ** This should be call before reading developing data or test data (just once at first time ).
    **/
    void add_special_flag_and_freeze_dict()
    {
        if (word_dict.is_frozen() && postag_dict.is_frozen() && ner_dict.is_frozen()) return; // has been frozen
        // First , add flag for start , end of word sequence and start of tag sequence  
        //SOS = word_dict_wrapper.Convert(SOS_STR);
        //EOS = word_dict_wrapper.Convert(EOS_STR);
        //SOS_TAG = tag_dict.Convert(SOS_TAG_STR); // The SOS_TAG should just using in tag hidden layer , and never in output ! 
        // Freeze
        word_dict_wrapper.Freeze();
        postag_dict.Freeze();
        ner_dict.Freeze();
        // set unk
        word_dict_wrapper.SetUnk(UNK_STR);
        // tag dict do not set unk . if has unkown tag , we think it is the dataset error and  should be fixed .
        UNK = word_dict_wrapper.Convert(UNK_STR); // get unk id at model (may be usefull for debugging)

        WORD_DICT_SIZE = word_dict.size();
        POSTAG_DICT_SIZE = postag_dict.size();
        NER_DICT_SIZE = ner_dict.size();
        NER_LAYER_OUTPUT_DIM = ner_dict.size();
    }

    void finish_read_training_data() { add_special_flag_and_freeze_dict(); }

    void print_model_info()
    {
        cout << "------------Model structure info-----------\n"
            << "vocabulary size : " << WORD_DICT_SIZE << " with dimension : " << WORD_EMBEDDING_DIM << "\n"
            << "postag dict size : " << POSTAG_DICT_SIZE << " with dimension : " << POSTAG_EMBEDDING_DIM << "\n"
            << "ner dict size : " << NER_DICT_SIZE << " with dimension : " << NER_TAG_EMBEDDING_DIM << "\n"
            << "LSTM has layer : " << LSTM_LAYER << " , with x dimension : " << LSTM_X_DIM << " , h dim : " << LSTM_H_DIM << "\n" 
            << "ner tag hidden dim : " << NER_LAYER_HIDDEN_DIM << "\n"
            << "ner tag output dim : " << NER_LAYER_OUTPUT_DIM << "\n"
            << "--------------------------------------------" << endl;
    }

    void set_model_param_from_varmap(const po::variables_map& conf)
    {
        WORD_EMBEDDING_DIM = conf["word_embedding_dim"].as<unsigned>();
        POSTAG_EMBEDDING_DIM = conf["postag_embedding_dim"].as<unsigned>();
        LSTM_X_DIM = conf["lstm_x_dim"].as<unsigned>();
        LSTM_H_DIM = conf["lstm_h_dim"].as<unsigned>();
        LSTM_LAYER = conf["lstm_layer"].as<unsigned>();
        NER_TAG_EMBEDDING_DIM = conf["ner_embedding_dim"].as<unsigned>();
        NER_LAYER_HIDDEN_DIM = conf["ner_hidden_dim"].as<unsigned>();
    }

    void build_model_structure(bool is_print_model_info = true)
    {
        if (!ner_dict.is_frozen() || !word_dict.is_frozen() || !postag_dict.is_frozen()) {
            BOOST_LOG_TRIVIAL(error) << "`finish_read_training_data` should be call before build model structure \n Exit!";
            abort();
        }

        m = new Model();
        bilstm_layer = new BILSTMLayer(m, LSTM_LAYER, LSTM_X_DIM, LSTM_H_DIM);

        words_lookup_param = m->add_lookup_parameters(WORD_DICT_SIZE, { WORD_EMBEDDING_DIM }); // ADD for PRE_TAG
        postags_lookup_param = m->add_lookup_parameters(POSTAG_DICT_SIZE , { POSTAG_EMBEDDING_DIM }); // having `SOS_TAG`
        nertags_lookup_param = m->add_lookup_parameters(NER_DICT_SIZE, { NER_TAG_EMBEDDING_DIM });
        
        input_merge_layer = new Merge2Layer(m, WORD_EMBEDDING_DIM, POSTAG_EMBEDDING_DIM, LSTM_X_DIM);

        bilstm_pretag_merge_layer = new Merge3Layer(m, LSTM_H_DIM, LSTM_H_DIM, NER_TAG_EMBEDDING_DIM, NER_LAYER_HIDDEN_DIM);

        pretag_SOS_param = m->add_parameters({ NER_TAG_EMBEDDING_DIM });

        output_linear_layer = new DenseLayer(m, NER_LAYER_HIDDEN_DIM, NER_LAYER_OUTPUT_DIM);

        if (is_print_model_info) print_model_info();
    }


    void load_wordembedding_from_word2vec_txt_format(istream &is)
    {
        // set lookup parameters from outer word embedding
        // using words_loopup_param.Initialize( word_id , value_vector )
        string line;
        vector<string> split_cont;
        split_cont.reserve(WORD_EMBEDDING_DIM + 1); // word + numbers 
        getline(is, line); // first line is the infomation !
        boost::split(split_cont, line, boost::is_any_of(" "));
        assert(2 == split_cont.size() && stoul(split_cont.at(1)) == WORD_EMBEDDING_DIM);
        unsigned long long line_cnt = 1 ;
        unsigned long long words_cnt_hit = 0;
        vector<float> embedding_vec(WORD_EMBEDDING_DIM , 0.f);
        while (getline(is, line))
        {
            ++line_cnt;
            boost::trim_right(line);
            boost::split(split_cont, line , boost::is_any_of(" "));
            if (WORD_EMBEDDING_DIM + 1 != split_cont.size())
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
        to << WORD_EMBEDDING_DIM << WORD_DICT_SIZE
            << POSTAG_EMBEDDING_DIM << POSTAG_DICT_SIZE
            << NER_TAG_EMBEDDING_DIM << NER_DICT_SIZE
            << LSTM_X_DIM << LSTM_H_DIM << LSTM_LAYER
            << NER_LAYER_HIDDEN_DIM << NER_LAYER_OUTPUT_DIM;

        to << word_dict << postag_dict << ner_dict ;
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
        ti >> WORD_EMBEDDING_DIM >> WORD_DICT_SIZE
            >> POSTAG_EMBEDDING_DIM >> POSTAG_DICT_SIZE
            >> NER_TAG_EMBEDDING_DIM >> NER_DICT_SIZE
            >> LSTM_X_DIM >> LSTM_H_DIM >> LSTM_LAYER
            >> NER_LAYER_HIDDEN_DIM >> NER_LAYER_OUTPUT_DIM;

        ti >> word_dict >> postag_dict >> ner_dict ;

        assert(WORD_DICT_SIZE == word_dict.size() && POSTAG_DICT_SIZE == postag_dict.size()
               && NER_DICT_SIZE == ner_dict.size());
       
        // SET VALUE for special flag
        //SOS = word_dict.Convert(SOS_STR);
        //EOS = word_dict.Convert(EOS_STR);
        //SOS_TAG = tag_dict.Convert(SOS_TAG_STR); // The SOS_TAG should just using in tag hidden layer , and never in output ! 
        UNK = word_dict.Convert(UNK_STR); // get unk id at model (may be usefull for debugging)

        // 2. build model structure 
        build_model_structure();

        // 3. load model parameter
        ti >> *m; // It is strange ! when I try to deserialize the pointer , it crashed ! using Entity avoid it !
              // It should be attention !

        BOOST_LOG_TRIVIAL(info) << "load model done .";
    }

    /****************************Train , Devel , Predict*********************************/

    Expression negative_loglikelihood(const IndexSeq *p_sent, const IndexSeq *p_postag_seq, const IndexSeq *p_ner_seq , 
        ComputationGraph *p_cg, Stat *p_stat = nullptr)
    {
        const unsigned sent_len = p_sent->size();
        ComputationGraph &cg = *p_cg;
        // New graph , ready for new sentence 
        bilstm_layer->new_graph(cg);
        input_merge_layer->new_graph(cg);
        bilstm_pretag_merge_layer->new_graph(cg);
        output_linear_layer->new_graph(cg);

        bilstm_layer->start_new_sequence();
        // Add parameters to cg
        // - ner layer hidden
        Expression pretag_SOS_exp = parameter(cg, pretag_SOS_param);


        // Some container
        vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
        vector<Expression> input_affine_exp_cont(sent_len); // for storing input exp ( `input_word_w * word_embedding + input_postag_w * postag_embedding + input_b` at every pos )
        vector<Expression> l2r_lstm_output_exp_cont(sent_len); // for storing left to right lstm output(deepest hidden layer) expression for every timestep
        vector<Expression> r2l_lstm_output_exp_cont(sent_len); // right to left 
        vector<Expression> pretag_lookup_exp_cont(sent_len); // ADD for PRE_TAG

        // build computation graph(also get output expression) for left to right LSTM
        // 1. get word embeddings for sent , postag embedding for postag_seq
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
            Expression postag_lookup_exp = lookup(cg, postags_lookup_param, p_postag_seq->at(i));
            Expression input_affine_exp = input_merge_layer->build_graph(word_lookup_exp, postag_lookup_exp);
            input_affine_exp_cont[i] = noise(input_affine_exp, 0.1f);
        }


        // 2. calc Expression of every timestep of BI-LSTM
        bilstm_layer->build_graph(input_affine_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
        // 3. prepare for PRE_TAG embedding
        pretag_lookup_exp_cont[0] = pretag_SOS_exp ;
        for (unsigned i = 1; i < sent_len ; ++i)
        {
            pretag_lookup_exp_cont[i] = lookup(cg, nertags_lookup_param, p_ner_seq->at(i - 1));
        }

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            // rectify is suggested as activation function
            Expression bilstm_pretag_merge_exp = bilstm_pretag_merge_layer->build_graph(l2r_lstm_output_exp_cont[i],
                r2l_lstm_output_exp_cont[i], pretag_lookup_exp_cont[i]);
            Expression tag_hidden_layer_output_at_timestep_t = cnn::expr::rectify(bilstm_pretag_merge_exp); // ADD for PRE_TAG
            
            Expression tag_output_layer_output_at_timestep_t = output_linear_layer->build_graph(tag_output_layer_output_at_timestep_t);
            
            // if statistic , calc output at timestep t
            if (p_stat != nullptr)
            {
                vector<float> output_values = as_vector(cg.incremental_forward());
                float max_value = output_values[0];
                Index tag_id_with_max_value = 0;
                for (unsigned i = 1; i < NER_LAYER_OUTPUT_DIM; ++i)
                {
                    if (max_value < output_values[i])
                    {
                        max_value = output_values[i];
                        tag_id_with_max_value = i;
                    }
                }
                ++(p_stat->total_tags); // == ++stat->total_tags ;
                if (tag_id_with_max_value == p_ner_seq->at(i)) ++(p_stat->correct_tags);
            }
            err_exp_cont[i] = pickneglogsoftmax(tag_output_layer_output_at_timestep_t, p_ner_seq->at(i));
        }

        // build the finally loss 
        return sum(err_exp_cont); // in fact , no need to return . just to avoid a warning .
    }

    void do_predict(const IndexSeq *p_sent, const IndexSeq *p_postag_seq, IndexSeq *p_predict_tag_seq , ComputationGraph *p_cg )
    {
        // The main structure is just a copy from build_bilstm4tagging_graph2train! 
        const unsigned sent_len = p_sent->size();
        ComputationGraph &cg = *p_cg;
        // New graph , ready for new sentence 
        bilstm_layer->new_graph(cg);
        input_merge_layer->new_graph(cg);
        bilstm_pretag_merge_layer->new_graph(cg);
        output_linear_layer->new_graph(cg);

        bilstm_layer->start_new_sequence();

        // Add parameters to cg
        Expression pretag_SOS_exp = parameter(cg, pretag_SOS_param);

        // Some container
        vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
        vector<Expression> input_affine_exp_cont(sent_len); // for storing input exp ( `input_word_w * word_embedding + input_postag_w * postag_embedding + input_b` at every pos )
        vector<Expression> l2r_lstm_output_exp_cont(sent_len); // for storing left to right lstm output(deepest hidden layer) expression for every timestep
        vector<Expression> r2l_lstm_output_exp_cont(sent_len); // right to left 
        vector<Expression> pretag_lookup_exp_cont(sent_len); // ADD for PRE_TAG

        // build computation graph(also get output expression) for left to right LSTM
        // 1. get word embeddings for sent 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
            Expression postag_lookup_exp = lookup(cg, postags_lookup_param, p_postag_seq->at(i));
            input_affine_exp_cont[i] = input_merge_layer->build_graph(word_lookup_exp , postag_lookup_exp);
        }


        // 2 calc Expression of every timestep of BI-LSTM

        bilstm_layer->build_graph(input_affine_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
        // 3. set previous tag lookup expression

        Expression pretag_lookup_exp = pretag_SOS_exp ;

        // build tag network , calc loss Expression of every timestep 
        for (unsigned i = 0; i < sent_len; ++i)
        {
            Expression bilstm_pretag_merge_exp = bilstm_pretag_merge_layer->build_graph(l2r_lstm_output_exp_cont[i],
                r2l_lstm_output_exp_cont[i], pretag_lookup_exp);
            Expression tag_hidden_layer_output_at_timestep_t = cnn::expr::rectify(bilstm_pretag_merge_exp);
            output_linear_layer->build_graph(tag_hidden_layer_output_at_timestep_t); 
            vector<float> output_values = as_vector(cg.incremental_forward());
            float max_value = output_values[0];
            unsigned tag_id_with_max_value = 0;
            for (unsigned i = 1; i < NER_LAYER_OUTPUT_DIM; ++i)
            {
                if (max_value < output_values[i])
                {
                    max_value = output_values[i];
                    tag_id_with_max_value = i;
                }
            }
            p_predict_tag_seq->push_back(tag_id_with_max_value);
            // set pretag_lookup_exp for next timestep 
            pretag_lookup_exp = lookup(cg, nertags_lookup_param, tag_id_with_max_value);
        }
    }

    void train(const vector<IndexSeq> *p_sents, const vector<IndexSeq> *p_postag_seqs , const vector<IndexSeq> *p_ner_seqs ,
        unsigned max_epoch, const vector<IndexSeq> *p_dev_sents = nullptr, const vector<IndexSeq> *p_dev_postag_seqs = nullptr ,
        const vector<IndexSeq> *p_dev_ner_seqs = nullptr ,
        const string conlleval_script_path="./ner_eval.sh" ,
        const unsigned long do_devel_freq=10000)
    {
        unsigned nr_samples = p_sents->size();

        BOOST_LOG_TRIVIAL(info) << "Train at " << nr_samples << " instances .\n";

        vector<unsigned> access_order(nr_samples);
        for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;

        SimpleSGDTrainer sgd = SimpleSGDTrainer(m);
        unsigned long line_cnt_for_devel = 0;
        unsigned long long total_time_cost_in_seconds = 0ULL ;
        IndexSeq sent_unk_replace;
        sent_unk_replace.reserve(256);
        for (unsigned nr_epoch = 0; nr_epoch < max_epoch; ++nr_epoch)
        {
            // shuffle samples by random access order
            shuffle(access_order.begin(), access_order.end(), *rndeng);

            // For loss , accuracy , time cost report
            Stat training_stat_per_report, training_stat_per_epoch;
            unsigned report_freq = 5000;

            // training for an epoch
            training_stat_per_report.start_time_stat();
            training_stat_per_epoch.start_time_stat() ;
            for (unsigned i = 0; i < nr_samples; ++i)
            {
                size_t access_idx = access_order[i];
                const IndexSeq &sent = p_sents->at(access_idx);
                const IndexSeq &postag_seq = p_postag_seqs->at(access_idx);
                const IndexSeq &ner_seq = p_ner_seqs->at(access_idx);
                
                // replace low-frequence word(idx) -> UNK in probability 
                sent_unk_replace.resize(sent.size());
                for (IndexSeq::const_iterator ite = sent.cbegin(); ite != sent.cend(); ++ite)
                {
                    sent_unk_replace[ite - sent.cbegin()] = word_dict_wrapper.ConvertProbability(*ite);
                }
                // using negative_loglikelihood loss to build model
                ComputationGraph *cg = new ComputationGraph(); // because at one scope , only one ComputationGraph is permited .
                                                               // so we have to declaring it as pointer and destroy it handly 
                                                               // before develing.
                negative_loglikelihood(&sent_unk_replace, &postag_seq, &ner_seq, cg, &training_stat_per_report);
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
                if (p_dev_sents != nullptr && 0 == line_cnt_for_devel % do_devel_freq)
                {
                    BOOST_LOG_TRIVIAL(info) << "do validation at every " << do_devel_freq << " samples " ;
                    float F1 = devel(p_dev_sents , p_dev_postag_seqs , p_dev_ner_seqs , conlleval_script_path);
                    if (F1 > best_F1)
                    {
                        BOOST_LOG_TRIVIAL(info) << "Better model found . stash it .";
                        best_F1 = F1;
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
                << nr_samples << " instances has been trained .\n"
                << "For this epoch , E = "
                << training_stat_per_epoch.get_E() << "\n"
                << "ACC = " << training_stat_per_epoch.get_acc() * 100 << " %\n"
                << "Total time cost " << epoch_time_cost << " s"
                << "Total tags : " << training_stat_per_epoch.total_tags << " , correct tags : " << training_stat_per_epoch.correct_tags << "\n"
                << "Current best F1 score " << best_F1 << "\n";

            total_time_cost_in_seconds += training_stat_per_epoch.get_time_cost_in_seconds();
            // do devel at every end of Epoch
            if (p_dev_sents != nullptr)
            {
                BOOST_LOG_TRIVIAL(info) << "do validation at every ends of epoch ." ;
                float F1 = devel(p_dev_sents , p_dev_postag_seqs , p_dev_ner_seqs , conlleval_script_path);
                if (F1 > best_F1)
                {
                    BOOST_LOG_TRIVIAL(info) << "Better model found . stash it .";
                    best_F1 = F1;
                    best_model_tmp_ss.str(""); // first , clear it's content !
                    boost::archive::text_oarchive to(best_model_tmp_ss);
                    to << *m;
                }
            }
        }
        BOOST_LOG_TRIVIAL(info) << "Training finished with cost " << total_time_cost_in_seconds << " s .";
    }

    float devel(const vector<IndexSeq> *p_dev_sents , const vector<IndexSeq> *p_dev_postag_seqs , 
        const vector<IndexSeq> *p_dev_ner_seqs , const string conlleval_script_path="./ner_eval.sh" )
    {
        unsigned nr_samples = p_dev_sents->size();
        BOOST_LOG_TRIVIAL(info) << "validation at " << nr_samples << " instances .";
        unsigned long line_cnt4error_output = 0;
        NerStat stat(conlleval_script_path);
        stat.start_time_stat();
        vector<IndexSeq> predict_ner_seqs(*p_dev_ner_seqs); // copy
        for (size_t idx = 0; idx < nr_samples; ++idx )
        {

            ++line_cnt4error_output;
            ComputationGraph cg;
            IndexSeq predict_ner_seq;
            const IndexSeq &sent = p_dev_sents->at(idx),
                &postag_seq = p_dev_postag_seqs->at(idx) ,
                &ner_seq = p_dev_ner_seqs->at(idx);
            do_predict(&sent, &postag_seq, &predict_ner_seq, &cg);
            assert(predict_ner_seq.size() == ner_seq.size());
            predict_ner_seqs[idx] = predict_ner_seq;
        }
        stat.end_time_stat();
        float F1 = stat.conlleval(*p_dev_ner_seqs , predict_ner_seqs , ner_dict);
        BOOST_LOG_TRIVIAL(info) << "validation finished . F1 = "
            << F1
            << ", with time cosing " << stat.get_time_cost_in_seconds() << " s . ";
        return F1 ;
    }

    void predict(istream &is, ostream &os)
    {
        const string SPLIT_DELIMITER = "\t";
        vector<vector<string>> raw_sents;
        vector<IndexSeq> sents,
            postag_seqs;
        read_test_data(is, &raw_sents, &sents , &postag_seqs);
        assert(raw_sents.size() == sents.size());
        BOOST_LOG_TRIVIAL(info) << "do prediction on " << raw_sents.size() << " instances .";
        BasicStat stat;
        stat.start_time_stat();
        for (unsigned int i = 0; i < raw_sents.size(); ++i)
        {
            vector<string> *p_raw_sent = &raw_sents.at(i);
            if (0 == p_raw_sent->size())
            {
                os << "\n";
                continue;
            }
            IndexSeq *p_sent = &sents.at(i);
            IndexSeq *p_postag_seq = &postag_seqs.at(i);
            IndexSeq predict_ner_seq;
            ComputationGraph cg;
            do_predict(p_sent, p_postag_seq , &predict_ner_seq , &cg);
            // output the result directly
            os << p_raw_sent->at(0) 
                << "/" << postag_dict.Convert(p_postag_seq->at(0)) 
                << "#" << ner_dict.Convert(predict_ner_seq.at(0));
            for (unsigned k = 1; k < p_raw_sent->size(); ++k)
            {
                os << SPLIT_DELIMITER
                    << p_raw_sent->at(k) 
                    << "/" << postag_dict.Convert(p_postag_seq->at(k)) 
                    << "#" << ner_dict.Convert(predict_ner_seq.at(k));
            }
            os << "\n";
        }
        stat.end_time_stat();
        BOOST_LOG_TRIVIAL(info) << "predict finished , costing " << stat.get_time_cost_in_seconds() << " s.";
    }
};


const string BILSTMModel4NER::number_transform_str = "##";
const size_t BILSTMModel4NER::length_transform_str = number_transform_str.length();

} // End of namespace slnn 
