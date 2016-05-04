#include "bilstmcrf.h"
#include "bilstmcrf_modelhandler.h"

using namespace std;
using namespace slnn;
namespace po = boost::program_options;
const string PROGRAM_DESCRIPTION = "Postagger-BILSTM_CRF based on CNN Library";


int train_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Training process .\n"
        "using `" + program_name + " train <options>` to train . Training options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("training_data", po::value<string>(), "[required] The path to training data")
        ("devel_data", po::value<string>(), "The path to developing data . For validation duration training . Empty for discarding .")
        ("word2vec_embedding", po::value<string>(), "The path to word embedding . only support word2vec txt-mode output result . "
            "dimension should be consistent with parameter `input_dim`. Empty for using randomized initialization .")
        ("max_epoch", po::value<unsigned>()->default_value(4), "The epoch to iterate for training")
        ("devel_freq", po::value<unsigned>()->default_value(100000), "The frequent(samples number)to validate(if set) . validation will be done after every devel-freq training samples")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("dynamic_embedding_dim", po::value<unsigned>()->default_value(50), "The dimension for dynamic channel word embedding.")
        ("postag_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for postag embedding.")
        ("nr_lstm_stacked_layer", po::value<unsigned>()->default_value(1), "The number of stacked layers in bi-LSTM.")
        ("lstm_x_dim", po::value<unsigned>()->default_value(50) , "The dimension for LSTM X .")
        ("lstm_h_dim", po::value<unsigned>()->default_value(100), "The dimension for LSTM H.")
        ("merge_hidden_dim", po::value<unsigned>()->default_value(100), "The dimension for tag hidden layer.")
        ("replace_freq_threshold", po::value<unsigned>()->default_value(1), "The frequency threshold to replace the word to UNK in probability"
            "(eg , if set 1, the words of training data which frequency <= 1 may be "
            " replaced in probability)")
        ("replace_prob_threshold", po::value<float>()->default_value(0.2f), "The probability threshold to replace the word to UNK ."
                " if words frequency <= replace_freq_threshold , the word will"
                " be replace in this probability")
        ("logging_verbose", po::value<int>()->default_value(0), "The switch for logging trace . If 0 , trace will be ignored ,"
                    "else value leads to output trace info.")
        ("do_train_stat" , po::value<bool>()->default_value(false) , "Whether doing stat during traing ."
                    "default false in bilstm crf for decreasing time cost ")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }
    // trace switch
    if (0 == var_map.count("logging_verbose") || 0 == var_map["logging_verbose"].as<int>())
    {
        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= boost::log::trivial::debug
        );
    }
    // set params 
    string training_data_path, devel_data_path , embedding_path ;
    if (0 == var_map.count("training_data"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Error : Training data should be specified ! \n"
            "using `" + program_name + " train -h ` to see detail parameters .\n"
            "Exit .";
        return -1;
    }
    training_data_path = var_map["training_data"].as<string>();
    if (0 == var_map.count("devel_data")) devel_data_path = "";
    else devel_data_path = var_map["devel_data"].as<string>();
    if (0 == var_map.count("word2vec_embedding"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Error : word2vec embedding path should be specified ! \n"
            "using `" + program_name + " train -h ` to see detail parameters .\n"
            "Exit .";
        return -1;
    }
    embedding_path = var_map["word2vec_embedding"].as<string>();
    unsigned max_epoch = var_map["max_epoch"].as<unsigned>();
    unsigned devel_freq = var_map["devel_freq"].as<unsigned>();
    bool do_train_stat = var_map["do_train_stat"].as<bool>() ;
    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    // others will be processed flowing 

    // Init 
    cnn::Initialize(argc, argv, 1234); // 
    BILSTMCRFModel4POSTAG dc_model;
    BILSTMCRFModelHandler model_handler(dc_model);
    // reading traing data , get word dict size and output tag number
    // -> set replace frequency for word_dict_wrapper
    model_handler.set_unk_replace_threshold(replace_freq_threshold, replace_prob_threshold);
    // build fixed dict 
    ifstream embedding_is(embedding_path);
    if (!embedding_is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open word2vec embedding : `" << training_data_path << "` .\n Exit! \n";
        return -1;
    }
    model_handler.build_fixed_dict_from_word2vec_file(embedding_is);
    embedding_is.clear() ; // !! MUST calling before `seekg` ! even thouth using  c++ 11 .
    embedding_is.seekg(0 , embedding_is.beg); // will use in the following 

    ifstream train_is(training_data_path);
    if (!train_is) {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open training: `" << training_data_path << "` .\n Exit! \n";
        return -1;
    }
    vector<IndexSeq> dynamic_sents , 
        fixed_sents , 
        postag_seqs ;
    model_handler.read_training_data_and_build_dynamic_and_postag_dicts(train_is, dynamic_sents , fixed_sents , postag_seqs);
    train_is.close();
    // set model structure param 
    model_handler.finish_read_training_data(var_map);
    
    // build model structure
    model_handler.build_model(); // passing the var_map to specify the model structure

    // load fixed embedding 
    model_handler.load_fixed_embedding(embedding_is);
    embedding_is.close();
    
    // reading developing data
    vector<IndexSeq> dev_dynamic_sents, *p_dev_dynamic_sents ,
        dev_fixed_sents, *p_dev_fixed_sents ,
        dev_postag_seqs, *p_dev_postag_seqs;
    if ("" != devel_data_path)
    {
        std::ifstream devel_is(devel_data_path);
        if (!devel_is) {
            BOOST_LOG_TRIVIAL(error) << "failed to open devel file: `" << devel_data_path << "`\n Exit!";
            // if set devel data , but open failed , we exit .
            return -1;
        }
        model_handler.read_devel_data(devel_is, dev_dynamic_sents , dev_fixed_sents , dev_postag_seqs);
        devel_is.close();
        p_dev_dynamic_sents = &dev_dynamic_sents;
        p_dev_fixed_sents = &dev_fixed_sents;
        p_dev_postag_seqs = &dev_postag_seqs;
    }
    else
    {
        p_dev_dynamic_sents = p_dev_fixed_sents = p_dev_postag_seqs = nullptr;
    }

    // Train 
    model_handler.train(&dynamic_sents , &fixed_sents , &postag_seqs , max_epoch, 
        p_dev_dynamic_sents , p_dev_fixed_sents , p_dev_postag_seqs ,
        devel_freq , do_train_stat );

    // save model
    string model_path;
    if (0 == var_map.count("model"))
    {
        cerr << "no model name specified . using default .\n";
        ostringstream oss;
        oss << "dc_" << dc_model.dynamic_embedding_dim << "_" << dc_model.lstm_h_dim
            << "_" << dc_model.merge_hidden_dim << ".model";
        model_path = oss.str();
    }
    else model_path = var_map["model"].as<string>();
    ofstream os(model_path);
    if (!os)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open model path at '" << model_path << "'. \n Exit !";
        return -1;
    }
    model_handler.save_model(os);
    os.close();
    return 0;
}

int devel_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Validation(develop) process "
        "using `" + program_name + " devel <options>` to validate . devel options are as following";
    po::options_description op_des = po::options_description(description);
    // set params to receive the arguments 
    string devel_data_path, model_path, error_output_path;
    op_des.add_options()
        ("devel_data", po::value<string>(&devel_data_path), "The path to validation data .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)")
        ("error_output", po::value<string>(&error_output_path), "Specify the file path to storing the predict error infomation . Empty to discard.")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }

    if ("" == devel_data_path)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Validation(develop) data should be specified !\n"
            "Exit!";
        return -1;
    }
    if ("" == model_path)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Model path should be specified ! \n"
            "Exit! ";
        return -1;
    }

    // Init 
    cnn::Initialize(argc, argv, 1234);
    BILSTMCRFModel4POSTAG dc_model;
    BILSTMCRFModelHandler model_handler(dc_model);
    // Load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n"
            "Exit !";
        return -1;
    }
    model_handler.load_model(is);
    is.close();
    dc_model.print_model_info();

    // read validation(develop) data
    std::ifstream devel_is(devel_data_path);
    if (!devel_is) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open devel file: `" << devel_data_path << "`\n"
            "Exit! ";
        return -1;
    }

    // ready error output file
    ofstream *p_error_output_os = nullptr;
    if (0U != error_output_path.size())
    {
        p_error_output_os = new ofstream(error_output_path);
        if (!p_error_output_os)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to open error output file : `" << error_output_path << "`\nExit!";
            return -1;
        }
    }
    // read devel data
    vector<IndexSeq> dynamic_sents,
        fixed_sents,
        postag_seqs;
    model_handler.read_devel_data(devel_is, dynamic_sents , fixed_sents , postag_seqs);
    devel_is.close();

    // devel
    model_handler.devel(&dynamic_sents , &fixed_sents , &postag_seqs, p_error_output_os); // Get the same result , it is OK .
    if (p_error_output_os)
    {
        p_error_output_os->close();
        delete p_error_output_os;
    }
    return 0;
}


int predict_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Predict process ."
        "using `" + program_name + " predict <options>` to predict . predict options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("raw_data", po::value<string>(), "The path to raw data(It should be segmented) .")
        ("output", po::value<string>(), "The path to storing result . using `stdout` if not specified .")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }

    //set params 
    string raw_data_path, output_path, model_path;
    if (0 == var_map.count("raw_data"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "raw_data path should be specified .\n"
            "Exit!";
        return -1;
    }
    else raw_data_path = var_map["raw_data"].as<string>();

    if (0 == var_map.count("output"))
    {
        BOOST_LOG_TRIVIAL(info) << "no output is specified . using stdout .";
        output_path = "";
    }
    else output_path = var_map["output"].as<string>();

    if (0 == var_map.count("model"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Model path should be specified ! \n"
            "Exit! ";
        return -1;
    }
    else model_path = var_map["model"].as<string>();

    // Init 
    cnn::Initialize(argc, argv, 1234);
    BILSTMCRFModel4POSTAG dc_model;
    BILSTMCRFModelHandler model_handler(dc_model);



    // load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n"
            "Exit .";
        return -1;
    }
    model_handler.load_model(is);
    is.close();

    // open raw_data
    ifstream raw_is(raw_data_path);
    if (!raw_is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open raw data at '" << raw_data_path << "' \n Exit .";
        return -1;
    }

    // open output 
    if ("" == output_path)
    {
        model_handler.predict(raw_is, cout); // using `cout` as output stream 
        raw_is.close();
    }
    else
    {
        ofstream os(output_path);
        if (!os)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed open output file at : `" << output_path << "`\n Exit .";
            raw_is.close();
            return -1;
        }
        model_handler.predict(raw_is, os);
        os.close();
        is.close();
    }
    return 0;
}


int main(int argc, char *argv[])
{
    string usage = PROGRAM_DESCRIPTION + "\n"
        "usage : " + string(argv[0]) + " [ train | devel | predict ] <options> \n"
        "using  `" + string(argv[0]) + " [ train | devel | predict ] -h` to see details for specify task\n";
    if (argc <= 1)
    {
        cerr << usage;
        return -1;
    }
    else if (string(argv[1]) == "train") return train_process(argc - 1, argv + 1, argv[0]);
    else if (string(argv[1]) == "devel") return devel_process(argc - 1, argv + 1, argv[0]);
    else if (string(argv[1]) == "predict") return predict_process(argc - 1, argv + 1, argv[0]);
    else
    {
        cerr << "unknown mode : " << argv[1] << "\n"
            << usage;
        return -1;
    }
}
