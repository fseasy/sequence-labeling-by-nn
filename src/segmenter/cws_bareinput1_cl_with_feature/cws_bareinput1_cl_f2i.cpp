#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include "cws_bareinput1_cl_f2i_model.h"
#include "segmenter/model_handler/input1_with_feature_modelhandler_0628.hpp"
#include "utils/general.hpp"

using namespace std;
using namespace dynet;
using namespace slnn;
namespace po = boost::program_options;
static const string ProgramHeader = "CWS BareInput1-Classification F2I Procedure based on DyNet Library";
static const int CNNRandomSeed = 1234;

template <typename RNNDerived>
int train_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Training process .\n"
        "using `" + program_name + " train [rnn-type] <options>` to train . Training options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .")
        ("training_data", po::value<string>(), "[required] The path to training data")
        ("devel_data", po::value<string>(), "The path to developing data . For validation duration training . Empty for discarding .")
        ("max_epoch", po::value<unsigned>(), "The epoch to iterate for training")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("dropout_rate", po::value<float>(), "droupout rate for training (Only for bi-lstm)")
        ("devel_freq", po::value<unsigned>()->default_value(100000), "The frequent(samples number)to validate(if set) . validation will be done after every devel-freq training samples")
        ("trivial_report_freq", po::value<unsigned>()->default_value(5000), "Trace frequent during training process")
        ("replace_freq_threshold", po::value<unsigned>()->default_value(1), "The frequency threshold to replace the word to UNK in probability"
            "(eg , if set 1, the words of training data which frequency <= 1 may be "
            " replaced in probability)")
        ("replace_prob_threshold", po::value<float>()->default_value(0.2f), "The probability threshold to replace the word to UNK ."
                " if words frequency <= replace_freq_threshold , the word will"
                " be replace in this probability")
        ("word_embedding_dim", po::value<unsigned>()->default_value(50), "The dimension for dynamic channel word embedding.")
        ("start_here_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for start-here-word-max-len feature .")
        ("pass_here_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for pass-here-word-max-len feature .")
        ("end_here_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for end-here-word-max-len feature .")
        ("context_left_size", po::value<unsigned>()->default_value(1), "The left size for context feature")
        ("context_right_size", po::value<unsigned>()->default_value(1), "The right size for context feature")
        ("chartype_embedding_dim", po::value<unsigned>()->default_value(3), "The dimension for chartype feature.")
        ("nr_rnn_stacked_layer", po::value<unsigned>()->default_value(1), "The number of stacked layers in bi-rnn.")
        ("rnn_h_dim", po::value<unsigned>()->default_value(100), "The dimension for rnn H.")
        ("logging_verbose", po::value<int>()->default_value(0), "The switch for logging trace . If 0 , trace will be ignored ,"
            "else value leads to output trace info.")
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
    if (0 == var_map["logging_verbose"].as<int>())
    {
        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= boost::log::trivial::debug
        );
    }
    // checking requiring key 
    string training_data_path, devel_data_path ;
    varmap_key_fatal_check(var_map, "training_data",
        "Error : Training data should be specified ! \n"
        "using `" + program_name + " train -h ` to see detail parameters .");
    training_data_path = var_map["training_data"].as<string>();


    varmap_key_fatal_check(var_map, "devel_data",
        "Error : devel data should be specified ! \n"
        "using `" + program_name + " train -h ` to see detail parameters .");
    devel_data_path = var_map["devel_data"].as<string>();  

    varmap_key_fatal_check(var_map, "max_epoch",
        "Error : max epoch num should be specified .");
    unsigned max_epoch = var_map["max_epoch"].as<unsigned>();

    varmap_key_fatal_check(var_map , "dropout_rate" ,
        "Error : dropout rate should be specified .") ;

    // check model path
    string model_path;
    varmap_key_fatal_check(var_map, "model",
        "Error : model path should be specified .");
    model_path = var_map["model"].as<string>();
    if (FileUtils::exists(model_path))
    {
        fatal_error("Error : model file `" + model_path + "` has already exists .");
    }
    // some key which has default value
    unsigned devel_freq = var_map["devel_freq"].as<unsigned>();
    unsigned trivial_report_freq = var_map["trivial_report_freq"].as<unsigned>();
    // others will be processed flowing 

    // Init 
    int dynet_argc;
    shared_ptr<char *> dynet_argv;
    unsigned dynet_mem = 0 ;
    if( var_map.count("dynet-mem") != 0 ){ dynet_mem = var_map["dynet-mem"].as<unsigned>();}
    build_dynet_parameters(program_name, dynet_mem, dynet_argc, dynet_argv);
    char **dynet_argv_ptr = dynet_argv.get();
    dynet::initialize(dynet_argc, dynet_argv_ptr, CNNRandomSeed); 
    CWSInput1WithFeatureModelHandler<RNNDerived, CWSBareInput1CLF2IModel<RNNDerived>> model_handler;

    // pre-open model file, avoid fail after a long time training
    ofstream model_os(model_path);
    if( !model_os ) fatal_error("failed to open model path at '" + model_path + "'") ;
    
    model_handler.set_model_param_before_reading_training_data(var_map);

    // reading traing data , get word dict size and output tag number
    ifstream train_is(training_data_path);
    if (!train_is) {
        fatal_error("Error : failed to open training: `" + training_data_path + "` .");
    }
    vector<IndexSeq> sents ,
        tag_seqs;
    vector<CWSFeatureDataSeq> feature_seqs;
    model_handler.read_training_data(train_is, sents ,feature_seqs, tag_seqs);
    train_is.close();
    // set model structure param 
    model_handler.set_model_param_after_reading_training_data();

    // build model structure
    model_handler.build_model(); // passing the var_map to specify the model structure

                                 // reading developing data
    vector<IndexSeq> dev_sents, dev_tag_seqs ;
    vector<CWSFeatureDataSeq> dev_feature_seqs ;
    std::ifstream devel_is(devel_data_path);
    if (!devel_is) {
        fatal_error("Error : failed to open devel file: `" + devel_data_path + "`");
    }
    model_handler.read_devel_data(devel_is, dev_sents , dev_feature_seqs, dev_tag_seqs);
    devel_is.close();

    // Train 
    model_handler.train(sents, feature_seqs, tag_seqs , 
        max_epoch, 
        dev_sents, dev_feature_seqs, dev_tag_seqs , 
        devel_freq , 
        trivial_report_freq);

    // save model
    model_handler.save_model(model_os);
    model_os.close();
    return 0;
}

template <typename RNNDerived>
int devel_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Validation(develop) process "
        "using `" + program_name + " devel [rnn-type] <options>` to validate . devel options are as following";
    po::options_description op_des = po::options_description(description);
    // set params to receive the arguments 
    string devel_data_path, model_path ;
    op_des.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .")
        ("devel_data", po::value<string>(&devel_data_path), "The path to validation data .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(op_des).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if (var_map.count("help"))
    {
        cerr << op_des << endl;
        return 0;
    }

    varmap_key_fatal_check(var_map, "devel_data", "Error : validation(develop) data should be specified !");
    varmap_key_fatal_check(var_map, "model", "Error : model path should be specified !");
    if( !FileUtils::exists(devel_data_path) ) fatal_error("Error : failed to find devel data at `" + devel_data_path + "`") ;

    // Init 
    int dynet_argc;
    shared_ptr<char *> dynet_argv;
    unsigned dynet_mem = 0 ;
    if( var_map.count("dynet-mem") != 0 ){ dynet_mem = var_map["dynet-mem"].as<unsigned>();}
    build_dynet_parameters(program_name, dynet_mem, dynet_argc, dynet_argv);
    char **dynet_argv_ptr = dynet_argv.get();
    dynet::initialize(dynet_argc, dynet_argv_ptr, CNNRandomSeed); 
    CWSInput1WithFeatureModelHandler<RNNDerived, CWSBareInput1CLF2IModel<RNNDerived>> model_handler;
    // Load model 
    ifstream model_is(model_path);
    if (!model_is)
    {
        fatal_error("Error : failed to open model path at '" + model_path + "' .");
    }
    model_handler.load_model(model_is);
    model_is.close();

    // read devel data
    ifstream devel_is(devel_data_path) ;
    if( !devel_is ) fatal_error("Error : failed to open devel data at `" + devel_data_path + "`") ;
    vector<IndexSeq> sents,
        tag_seqs ;
    vector<CWSFeatureDataSeq> feature_seqs;
    model_handler.read_devel_data(devel_is, sents, feature_seqs, tag_seqs);
    devel_is.close();

    // devel
    model_handler.devel(sents , feature_seqs, tag_seqs); 

    return 0;
}

template <typename RNNDerived>
int predict_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Predict process ."
        "using `" + program_name + " predict [rnn-type] <options>` to predict . predict options are as following";
    po::options_description op_des = po::options_description(description);
    string raw_data_path, output_path, model_path;
    op_des.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .")
        ("raw_data", po::value<string>(&raw_data_path), "The path to raw data(It should be segmented) .")
        ("output", po::value<string>(&output_path), "The path to storing result . using `stdout` if not specified .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)")
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

    varmap_key_fatal_check(var_map, "raw_data", "raw_data path should be specified .");

    if (output_path == "")
    {
        BOOST_LOG_TRIVIAL(info) << "no output is specified . using stdout .";
    }

    varmap_key_fatal_check(var_map, "model", "Error : model path should be specified ! ");

    // Init 
    int dynet_argc;
    shared_ptr<char *> dynet_argv;
    unsigned dynet_mem = 0 ;
    if( var_map.count("dynet-mem") != 0 ){ dynet_mem = var_map["dynet-mem"].as<unsigned>();}
    build_dynet_parameters(program_name, dynet_mem, dynet_argc, dynet_argv);
    char **dynet_argv_ptr = dynet_argv.get();
    dynet::initialize(dynet_argc, dynet_argv_ptr, CNNRandomSeed); 
    CWSInput1WithFeatureModelHandler<RNNDerived, CWSBareInput1CLF2IModel<RNNDerived>> model_handler ;

    // load model 
    ifstream is(model_path);
    if (!is)
    {
        fatal_error("Error : failed to open model path at '" + model_path + "' . ");
    }
    model_handler.load_model(is);
    is.close();

    // open raw_data
    ifstream raw_is(raw_data_path);
    if (!raw_is)
    {
        fatal_error("Error : failed to open raw data at '" + raw_data_path + "'");
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
            raw_is.close();
            fatal_error("Error : failed open output file at : `" +  output_path + "`.");
        }
        model_handler.predict(raw_is, os);
        os.close();
    }
    return 0;
}


int main(int argc, char *argv[])
{
    ostringstream oss;
    string program_name = argv[0];
    oss << ProgramHeader << "\n"
        << "usage : " << program_name << " [task] [rnn-type] <options>" << "\n"
        << "task : [ train, devel, predict ] , anyone of the list is optional\n"
        << "rnn-type : [ rnn, lstm, gru] , \n"
        << "           rnn  : simple rnn implementation for RNN\n"
        << "           lstm : lstm implementation for RNN\n"
        << "           gru  : gru implementation for RNN\n"
        << "<options> : options for specific task and model .\n"
        << "            using '" << program_name << " [task] [rnn-type] -h' for details" ;
    string usage = oss.str();

    if (argc <= 3)
    {
        cerr << usage << "\n" ;
#if (defined(_WIN32)) && (defined(_DEBUG))
        system("pause");
#endif
        return -1;
    }
    string task = string(argv[1]);
    string rnn_type = string(argv[2]);
    int ret_status ;
    const string TrainTask = "train", DevelTask = "devel", PredictTask = "predict";
    const string SimpleRNNType = "rnn", LSTMType = "lstm", GRUType = "gru";
    function<void()> action_when_unknown_rnn_type = [&ret_status,&rnn_type]
    {
        cerr << "unknow rnn-type : '" << rnn_type << "'\n";
        ret_status = -1;
    } ;
    if( TrainTask == task )
    {
        if( SimpleRNNType == rnn_type ){ ret_status = train_process<SimpleRNNBuilder>(argc - 2, argv + 2, program_name); }
        else if( LSTMType == rnn_type ){ ret_status = train_process<LSTMBuilder>(argc - 2, argv + 2, program_name); }
        else if( GRUType == rnn_type ){ ret_status = train_process<GRUBuilder>(argc - 2, argv + 2, program_name); }
        else{ action_when_unknown_rnn_type(); }
    }
    else if( DevelTask == task )
    {
        if( SimpleRNNType == rnn_type ){ ret_status = devel_process<SimpleRNNBuilder>(argc - 2, argv + 2, program_name); }
        else if( LSTMType == rnn_type ){ ret_status = devel_process<LSTMBuilder>(argc - 2, argv + 2, program_name); }
        else if( GRUType == rnn_type ){ ret_status = devel_process<GRUBuilder>(argc - 2, argv + 2, program_name); }
        else { action_when_unknown_rnn_type(); }
    }
    else if( PredictTask == task )
    {
        if( SimpleRNNType == rnn_type ){ ret_status = predict_process<SimpleRNNBuilder>(argc - 2, argv + 2, program_name); }
        else if( LSTMType == rnn_type ){ ret_status = predict_process<LSTMBuilder>(argc - 2, argv + 2, program_name); }
        else if( GRUType == rnn_type ){ ret_status = predict_process<GRUBuilder>(argc - 2, argv + 2, program_name); }
        else { action_when_unknown_rnn_type() ; }
    }
    else
    {
        cerr << "unknown task : " << task << "\n"
            << usage;
        ret_status = -1;
    }
#if (defined(_WIN32)) && (_DEBUG)
    system("pause");
#endif
    return ret_status;
}

