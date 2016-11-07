#include <boost/program_options.hpp>
#include "cws_mlp_input1_instance.h"
#include "segmenter/cws_module/cws_general_modelhandler.h"
#include "utils/general.hpp"

using namespace std;
namespace po = boost::program_options;
using namespace slnn;
using namespace slnn::segmenter;
using slnn::segmenter::mlp_input1::MlpInput1Bigram;

static const string PROGRAM_HEADER = "segmenter Mlp-input1-bigram based on DyNet Library";
constexpr unsigned DEFAULT_RNG_SEED = 1234;

int train_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_HEADER + "\n"
        "Training process .\n"
        "using `" + program_name + " train [rnn-type] <options>` to train . Training options are as following";
    po::options_description generic_op("generic options");
    generic_op.add_options()
        ("config", po::value<string>(), "Path to config file")
        ("logging_verbose", po::value<int>()->default_value(0), "The switch for logging trace . If 0 , trace will be ignored ,"
            "else value leads to output trace info.")
        ("help,h", "Show help information.");
    
    po::options_description dynet_op("dynet options");
    dynet_op.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .");
    po::options_description file_op("file options");
    file_op.add_options()
        ("training_data", po::value<string>(), "[required] The path to training data")
        ("devel_data", po::value<string>(), "The path to developing data . For validation duration training . Empty for discarding .")
        ("model", po::value<string>(), "Use to specify the model name(path)");

    po::options_description training_op("training options");
    training_op.add_options()
        ("max_epoch", po::value<unsigned>(), "The epoch to iterate for training")
        ("devel_freq", po::value<unsigned>()->default_value(100000), "The frequent(samples number)to validate(if set) . validation will be done after every devel-freq training samples")
        ("training_update_scale", po::value<float>()->default_value(1.f), "The scale for backward updating.")
        ("training_update_method", po::value<string>()->default_value("sgd"), "The update method, support list: sgd, adagrad")
        ("trivial_report_freq", po::value<unsigned>()->default_value(5000), "Trace frequent during training process");

    po::options_description model_op("model options");
    model_op.add_options()
        ("rng_seed", po::value<unsigned>()->default_value(DEFAULT_RNG_SEED), "Random Number Generator seed.")
        ("word_embedding_dim", po::value<unsigned>()->default_value(50), "The dimension for dynamic channel word embedding.")
        ("window_size", po::value<unsigned>()->default_value(5), "The window size")
        ("window_process_method", po::value<string>()->default_value(string("concat")), "The method for window embedding processing."
            "support list: [concat, sum(avg), bigram]")
        ("mlp_hidden_dim_list",po::value<string>(), "The dimension list for mlp hidden layers , dims should be give positive number "
            "separated by comma , like : 512,256,334 ")
        ("dropout_rate", po::value<float>(), "droupout rate for training (mlp hidden layers)")
        ("nonlinear_func", po::value<string>()->default_value(string("relu")), "Non-linear function for mlp layers")
        ("output_layer_type", po::value<string>()->default_value(string("cl")), 
            "The output layer type, supporting list: [cl(classification), pretag, crf]")
        ("replace_freq_threshold", po::value<unsigned>()->default_value(1), "The frequency threshold to replace the word to UNK in probability"
            "(eg , if set 1, the words of training data which frequency <= 1 may be "
            "replaced in probability)")
        ("replace_prob_threshold", po::value<float>()->default_value(0.2f), "The probability threshold to replace the word to UNK ."
                " if words frequency <= replace_freq_threshold , the word will"
                " be replace in this probability");
    
    
    po::options_description all_op(description);
    all_op.add(generic_op).add(dynet_op).add(file_op).add(training_op).add(model_op);

    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(all_op).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if( var_map.count("config") )
    {
        ifstream conf_is(var_map["config"].as<string>());
        if( conf_is )
        {
            po::store(po::parse_config_file(conf_is, all_op, true),
                var_map);
            po::notify(var_map);
        }
        else
        {
            cerr << "failed to open config file:'" << var_map["config"].as<string>() <<"'\n";
        }
    }
    if (var_map.count("help"))
    {
        cerr << all_op << endl;
        return 0;
    }
    // trace switch
    if (0 == var_map["logging_verbose"].as<int>())
    {
        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= boost::log::trivial::debug
        );
    }
    // checking requiring key and build training options
    struct TrainingOpts
    {
        float training_update_scale;
        string training_update_method;
        unsigned do_devel_freq;
        unsigned max_epoch;
        unsigned trivial_report_freq;
    };
    TrainingOpts opts;
    string training_data_path, devel_data_path ;
    varmap_key_fatal_check(var_map, "training_data",
        "Error : Training data should be specified ! \n"
        "using `" + program_name + " train -h ` to see detail parameters .");
    training_data_path = var_map["training_data"].as<string>();
    varmap_key_fatal_check(var_map, "devel_data",
        "Error : devel data should be specified ! \n"
        "using `" + program_name + " train -h ` to see detail parameters .");
    devel_data_path = var_map["devel_data"].as<string>();  
    varmap_key_fatal_check(var_map , "mlp_hidden_dim_list" ,
        "Error : mlp hidden layer dims should be specified .") ;
    varmap_key_fatal_check(var_map , "dropout_rate" ,
        "Error : dropout rate should be specified .") ;

    varmap_key_fatal_check(var_map, "max_epoch",
        "Error : max epoch num should be specified .");
    opts.max_epoch = var_map["max_epoch"].as<unsigned>();
    opts.do_devel_freq = var_map["devel_freq"].as<unsigned>();
    opts.training_update_scale = var_map["training_update_scale"].as<float>();
    opts.training_update_method = var_map["training_update_method"].as<string>();
    opts.trivial_report_freq = var_map["trivial_report_freq"].as<unsigned>();
    // check model path
    string model_path;
    varmap_key_fatal_check(var_map, "model",
        "Error : model path should be specified .");
    model_path = var_map["model"].as<string>();
    if (FileUtils::exists(model_path))
    {
        fatal_error("Error : model file `" + model_path + "` has already exists .");
    }
    unsigned rng_seed = var_map["rng_seed"].as<unsigned>();
    // others will be processed flowing 

    // Init 
    //int dynet_argc;
    //shared_ptr<char *> dynet_argv;
    //unsigned dynet_mem = 0 ;
    //if( var_map.count("dynet-mem") != 0 ){ dynet_mem = var_map["dynet-mem"].as<unsigned>();}
    //build_dynet_parameters(program_name, dynet_mem, dynet_argc, dynet_argv);
    //char **dynet_argv_ptr = dynet_argv.get();
    std::shared_ptr<MlpInput1Bigram>  mi1 = MlpInput1Bigram::create_new_model(argc, argv, rng_seed);

    // pre-open model file, avoid fail after a long time training
    ofstream model_os(model_path);
    if( !model_os ) fatal_error("failed to open model path at '" + model_path + "'") ;

    mi1->set_model_structure_param_from_outer(var_map);
    // reading traing data , get word dict size and output tag number
    ifstream train_is(training_data_path);
    if (!train_is) {
        fatal_error("Error : failed to open training: `" + training_data_path + "` .");
    }

    vector<MlpInput1Bigram::AnnotatedDataProcessedT> training_data;
    modelhandler::read_training_data(train_is, *mi1, training_data);
    train_is.close();
    
    mi1->finish_read_training_data();
    // build model structure
    mi1->build_model_structure();

    // reading developing data
    vector<MlpInput1Bigram::AnnotatedDataProcessedT> devel_data;
    std::ifstream devel_is(devel_data_path);
    if (!devel_is) {
        fatal_error("Error : failed to open devel file: `" + devel_data_path + "`");
    }
    modelhandler::read_devel_data(devel_is, *mi1, devel_data);
    devel_is.close();
    
    // Train
    modelhandler::train(*mi1, training_data, devel_data, opts);
     
    // save model
    mi1->save_model(model_os);
    model_os.close();
    return 0;
}

int devel_process(int argc, char *argv[], const string &program_name)
{
    
    string description = PROGRAM_HEADER + "\n"
        "Validation(develop) process "
        "using `" + program_name + " devel [rnn-type] <options>` to validate . devel options are as following";
    po::options_description generic_op("generic options");
    generic_op.add_options()
        ("config", po::value<string>(), "Path to config file")
        ("help,h", "Show help information.");
    
    po::options_description dynet_op("dynet options");
    dynet_op.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .");
    
    string devel_data_path, model_path ;
    po::options_description file_op("file options");
    file_op.add_options()
        ("devel_data", po::value<string>(&devel_data_path), "The path to developing data . For validation duration training . Empty for discarding .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)");

    po::options_description all_op = po::options_description(description);
    // set params to receive the arguments 
    all_op.add(generic_op).add(dynet_op).add(file_op);
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(all_op).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if( var_map.count("config") )
    {
        ifstream conf_is(var_map["config"].as<string>());
        if( conf_is )
        {
            po::store(po::parse_config_file(conf_is, all_op, true),
                var_map);
            po::notify(var_map);
        }
        else
        {
            cerr << "failed to open config file:'" << var_map["config"].as<string>() <<"'\n";
        }
    }
    if (var_map.count("help"))
    {
        cerr << all_op << endl;
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
    
    // Load model 
    ifstream model_is(model_path);
    if (!model_is)
    {
        fatal_error("Error : failed to open model path at '" + model_path + "' .");
    }
    std::shared_ptr<MlpInput1Bigram> mi1 = MlpInput1Bigram::load_and_build_model(model_is, dynet_argc, dynet_argv_ptr);
    model_is.close();

    // read devel data
    ifstream devel_is(devel_data_path) ;
    if( !devel_is ) fatal_error("Error : failed to open devel data at `" + devel_data_path + "`") ;
    vector<MlpInput1Bigram::AnnotatedDataProcessedT> devel_data;
    modelhandler::read_devel_data(devel_is, *mi1, devel_data);
    devel_is.close();

    // devel
    modelhandler::devel(*mi1, devel_data); 
    return 0;
}

int predict_process(int argc, char *argv[], const string &program_name)
{
    string description = PROGRAM_HEADER + "\n"
        "Predict process ."
        "using `" + program_name + " predict [rnn-type] <options>` to predict . predict options are as following";
    
    po::options_description generic_op("generic options");
    generic_op.add_options()
        ("config", po::value<string>(), "Path to config file")
        ("help,h", "Show help information.");

    po::options_description dynet_op("dynet options");
    dynet_op.add_options()
        ("dynet-mem", po::value<unsigned>(), "pre-allocated memory pool for DyNet library (MB) .");

    string raw_data_path, output_path, model_path;
    po::options_description file_op("file options");
    file_op.add_options()
        ("input", po::value<string>(&raw_data_path), "The path to input data.")
        ("output", po::value<string>(&output_path), "The path to storing result . using `stdout` if not specified .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)");
    
    po::options_description all_op = po::options_description(description);
    // set params to receive the arguments 
    all_op.add(generic_op).add(dynet_op).add(file_op);
    po::variables_map var_map;
    po::store(po::command_line_parser(argc, argv).options(all_op).allow_unregistered().run(), var_map);
    po::notify(var_map);
    if( var_map.count("config") )
    {
        ifstream conf_is(var_map["config"].as<string>());
        if( conf_is )
        {
            po::store(po::parse_config_file(conf_is, all_op, true),
                var_map);
            po::notify(var_map);
        }
        else
        {
            cerr << "failed to open config file:'" << var_map["config"].as<string>() <<"'\n";
        }
    }
    
    if (var_map.count("help"))
    {
        cerr << all_op << endl;
        return 0;
    }

    //set params 

    varmap_key_fatal_check(var_map, "input", "input data path should be specified .");

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

    // load model 
    ifstream is(model_path);
    if (!is)
    {
        fatal_error("Error : failed to open model path at '" + model_path + "' . ");
    }
    shared_ptr<MlpInput1Bigram> mi1 = MlpInput1Bigram::load_and_build_model(is, dynet_argc, dynet_argv_ptr);
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
        modelhandler::predict(*mi1, raw_is, cout); // using `cout` as output stream 
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
        modelhandler::predict(*mi1, raw_is, os);
        os.close();
    }
    return 0;
}


int main(int argc, char *argv[])
{
    ostringstream oss;
    string program_name = argv[0];
    oss << PROGRAM_HEADER << "\n"
        << "usage     : " << program_name << " [task] [options]" << "\n\n"
        << "task      : [ train, devel, predict ] , one of the list is optional\n"
        << "options   : options for specific task and model .\n"
        << "            using '" << program_name << " [task] -h' for details" ;
    string usage = oss.str();

    if (argc <= 2)
    {
        cerr << usage << "\n" ;
#if (defined(_WIN32)) && (defined(_DEBUG))
        system("pause");
#endif
        return -1;
    }
    string task = string(argv[1]);
    int ret_status ;
    const string TrainTask = "train", DevelTask = "devel", PredictTask = "predict";
    if( TrainTask == task )
    {
        ret_status = train_process(argc - 1, argv + 1, program_name); 
    }
    else if( DevelTask == task )
    {
        ret_status = devel_process(argc - 1, argv + 1, program_name); 
    }
    else if( PredictTask == task )
    {
        ret_status = predict_process(argc - 1, argv + 1, program_name); 
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
