#include <sstream>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/gru.h"

#include "pos_input1_mlp_with_tag_model.h"
#include "postagger/model_handler/input1_mlp_modelhandler.h"
#include "utils/general.hpp"

using namespace std;
using namespace cnn;
using namespace slnn;
namespace po = boost::program_options;
static const string ProgramHeader = "Postagger Input1-MLP With Tag based on CNN Library";
static const int CNNRandomSeed = 1234;


int train_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Training process .\n"
        "using `" + program_name + " train [rnn-type] <options>` to train . Training options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("cnn-mem", po::value<unsigned>(), "pre-allocated memory pool for CNN library (MB) .")
        ("training_data", po::value<string>(), "[required] The path to training data")
        ("devel_data", po::value<string>(), "The path to developing data . For validation duration training . Empty for discarding .")
        ("max_epoch", po::value<unsigned>(), "The epoch to iterate for training")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("devel_freq", po::value<unsigned>()->default_value(100000), "The frequent(samples number)to validate(if set) . validation will be done after every devel-freq training samples")
        ("trivial_report_freq", po::value<unsigned>()->default_value(5000), "Trace frequent during training process")
        ("replace_freq_threshold", po::value<unsigned>()->default_value(1), "The frequency threshold to replace the word to UNK in probability"
            "(eg , if set 1, the words of training data which frequency <= 1 may be "
            " replaced in probability)")
        ("replace_prob_threshold", po::value<float>()->default_value(0.2f), "The probability threshold to replace the word to UNK ."
                " if words frequency <= replace_freq_threshold , the word will"
                " be replace in this probability")
        ("word_embedding_dim", po::value<unsigned>()->default_value(50), "The dimension for word embedding.")
        ("tag_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for tag embedding")
        ("mlp_hidden_dim_list", po::value<string>(), "The dimension list for mlp hidden layers , dims should be give positive number "
            "separated by comma , like : 512,256,334 ")
        ("mlp_nonlinear_func", po::value<string>()->default_value(string("tanh")), 
            "nonlinear function for mlp hidden layer , only support tanh , rectify")
        ("dropout_rate", po::value<float>(), "droupout rate for training (mlp hidden layers)")
        ("prefix_suffix_len1_embedding_dim", po::value<unsigned>()->default_value(20), "The dimension for prefix suffix len1 feature .")
        ("prefix_suffix_len2_embedding_dim", po::value<unsigned>()->default_value(40), "The dimension for prefix suffix len2 feature .")
        ("prefix_suffix_len3_embedding_dim", po::value<unsigned>()->default_value(40), "The dimension for prefix suffix len3 feature .")
        ("char_length_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for character length feature .")
        ("context_left_size", po::value<unsigned>()->default_value(2), "The left size for context feature")
        ("context_right_size", po::value<unsigned>()->default_value(2), "The right size for context feature")
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

    varmap_key_fatal_check(var_map , "mlp_hidden_dim_list" ,
        "Error : mlp hidden layer dims should be specified .") ;
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
    unsigned devel_freq = var_map["devel_freq"].as<unsigned>();
    unsigned trivial_report_freq = var_map["trivial_report_freq"].as<unsigned>();
    // others will be processed flowing 

    // Init 
    int cnn_argc;
    shared_ptr<char *> cnn_argv;
    unsigned cnn_mem = 0 ;
    if( var_map.count("cnn-mem") != 0 ){ cnn_mem = var_map["cnn-mem"].as<unsigned>();}
    build_cnn_parameters(program_name, cnn_mem, cnn_argc, cnn_argv);
    char **cnn_argv_ptr = cnn_argv.get();
    cnn::Initialize(cnn_argc, cnn_argv_ptr, CNNRandomSeed); 
    Input1MLPModelHandler<POSInput1MLPWithTagModel> model_handler;

    // pre-open model file, avoid fail after a long time training
    ofstream model_os(model_path);
    if( !model_os ) fatal_error("failed to open model path at '" + model_path + "'") ;
    model_handler.set_model_param_before_read_training_data(var_map);
    // reading traing data , get word dict size and output tag number

    ifstream train_is(training_data_path);
    if (!train_is) {
        fatal_error("Error : failed to open training: `" + training_data_path + "` .");
    }
    vector<IndexSeq> sents ,
        tag_seqs;
    vector<ContextFeatureDataSeq> context_feature_gp_seqs;
    vector<POSFeature::POSFeatureIndexGroupSeq> feature_gp_seqs;
    model_handler.read_training_data(train_is, sents ,context_feature_gp_seqs, feature_gp_seqs, tag_seqs);
    train_is.close();
    // set model structure param 
    model_handler.set_model_param_after_read_training_data();

    // build model structure
    model_handler.build_model(); // passing the var_map to specify the model structure

                                 // reading developing data
    vector<IndexSeq> dev_sents, dev_tag_seqs ;
    vector<ContextFeatureDataSeq> dev_context_feature_gp_seqs;
    vector<POSFeature::POSFeatureIndexGroupSeq> dev_feature_gp_seqs ;
    std::ifstream devel_is(devel_data_path);
    if (!devel_is) {
        fatal_error("Error : failed to open devel file: `" + devel_data_path + "`");
    }
    model_handler.read_devel_data(devel_is, dev_sents, dev_context_feature_gp_seqs, dev_feature_gp_seqs, dev_tag_seqs);
    devel_is.close();

    // Train 
    model_handler.train(sents, context_feature_gp_seqs, feature_gp_seqs, tag_seqs , 
        max_epoch, 
        dev_sents , dev_context_feature_gp_seqs, dev_feature_gp_seqs, dev_tag_seqs , 
        devel_freq , 
        trivial_report_freq);

    // save model
    model_handler.save_model(model_os);
    model_os.close();
    return 0;
}

int devel_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Validation(develop) process "
        "using `" + program_name + " devel [rnn-type] <options>` to validate . devel options are as following";
    po::options_description op_des = po::options_description(description);
    // set params to receive the arguments 
    string devel_data_path, model_path ;
    op_des.add_options()
        ("cnn-mem", po::value<unsigned>(), "pre-allocated memory pool for CNN library (MB) .")
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
    int cnn_argc;
    shared_ptr<char *> cnn_argv;
    unsigned cnn_mem = 0 ;
    if( var_map.count("cnn-mem") != 0 ){ cnn_mem = var_map["cnn-mem"].as<unsigned>();}
    build_cnn_parameters(program_name, cnn_mem, cnn_argc, cnn_argv);
    char **cnn_argv_ptr = cnn_argv.get();
    cnn::Initialize(cnn_argc, cnn_argv_ptr, CNNRandomSeed); 
    Input1MLPModelHandler<POSInput1MLPWithTagModel> model_handler;
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
    vector<ContextFeatureDataSeq> context_feature_gp_seqs;
    vector<POSFeature::POSFeatureIndexGroupSeq> feature_gp_seqs;
    model_handler.read_devel_data(devel_is, sents, context_feature_gp_seqs, feature_gp_seqs, tag_seqs);
    devel_is.close();

    // devel
    model_handler.devel(sents , context_feature_gp_seqs, feature_gp_seqs, tag_seqs); 

    return 0;
}

int predict_process(int argc, char *argv[], const string &program_name)
{
    string description = ProgramHeader + "\n"
        "Predict process ."
        "using `" + program_name + " predict [rnn-type] <options>` to predict . predict options are as following";
    po::options_description op_des = po::options_description(description);
    string raw_data_path, output_path, model_path;
    op_des.add_options()
        ("cnn-mem", po::value<unsigned>(), "pre-allocated memory pool for CNN library (MB) .")
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
    int cnn_argc;
    shared_ptr<char *> cnn_argv;
    unsigned cnn_mem = 0 ;
    if( var_map.count("cnn-mem") != 0 ){ cnn_mem = var_map["cnn-mem"].as<unsigned>();}
    build_cnn_parameters(program_name, cnn_mem, cnn_argc, cnn_argv);
    char **cnn_argv_ptr = cnn_argv.get();
    cnn::Initialize(cnn_argc, cnn_argv_ptr, CNNRandomSeed); 
    Input1MLPModelHandler<POSInput1MLPWithTagModel> model_handler ;

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
        << "usage : " << program_name << " [task] <options>" << "\n"
        << "task : [ train, devel, predict ] , anyone of the list is optional\n"
        << "<options> : options for specific task and model .\n"
        << "            using '" << program_name << " [task] [rnn-type] -h' for details" ;
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
