#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options.hpp>

#include "bilstmmodel4tagging.hpp"

using namespace std;
using namespace cnn;
using namespace slnn;
namespace po = boost::program_options;



const string PROGRAM_DESCRIPTION = "Postagger based on CNN Library";

/********************************DEBUG OUPTUT FUNCTION***********************************/

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


/********************************Action****************************************/

int train_process(int argc, char *argv[] , const string &program_name) 
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Training process .\n"
        "using `" + program_name + " train <options>` to train . Training options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("training_data", po::value<string>(), "[required] The path to training data")
        ("devel_data", po::value<string>(), "The path to developing data . For validation duration training . Empty for discarding .")
        ("word_embedding", po::value<string>(), "The path to word embedding . support word2vec txt-mode output result . "
            "dimension should be consistent with parameter `input_dim`. Empty for using randomized initialization .")
        ("max_epoch", po::value<unsigned>()->default_value(4), "The epoch to iterate for training")
        ("devel_freq", po::value<unsigned long>()->default_value(100000), "The frequent(samples number)to validate(if set) . validation will be done after every devel-freq training samples")
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("input_dim", po::value<unsigned>()->default_value(50), "The dimension for input word embedding.")
        ("tag_embedding_dim", po::value<unsigned>()->default_value(5), "The dimension for tag embedding.")
        ("lstm_layers", po::value<unsigned>()->default_value(1), "The number of layers in bi-LSTM.")
        ("lstm_hidden_dim", po::value<unsigned>()->default_value(100), "The dimension for LSTM output.")
        ("tag_dim", po::value<unsigned>()->default_value(32), "The dimension for tag.")
        ("replace_freq_threshold" , po::value<unsigned>()->default_value(1) , "The frequency threshold to replace the word to UNK in probability"
                                                                               "(eg , if set 1, the words of training data which frequency <= 1 may be "
                                                                               " replaced in probability)")
        ("replace_prob_threshold" , po::value<float>()->default_value(0.2f) , "The probability threshold to replace the word to UNK ."
                                                                             " if words frequency <= replace_freq_threshold , the word will "
                                                                             "be replace in this probability")
        ("logging_verbose", po::value<int>()->default_value(0), "The switch for logging trace . If 0 , trace will be ignored ,"
                                                                "else value leads to output trace info.")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc,argv).options(op_des).allow_unregistered().run(), var_map);
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
    string training_data_path, devel_data_path;
    if (0 == var_map.count("training_data"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "Error : Training data should be specified ! \n"
            "using `-h` to see detail parameters .\n"
            "Exit .";
        return -1;
    }
    training_data_path = var_map["training_data"].as<string>();
    if (0 == var_map.count("devel_data")) devel_data_path = "";
    else devel_data_path = var_map["devel_data"].as<string>();
    unsigned max_epoch = var_map["max_epoch"].as<unsigned>();
    unsigned long devel_freq = var_map["devel_freq"].as<unsigned long>();
    unsigned replace_freq_threshold = var_map["replace_freq_threshold"].as<unsigned>();
    float replace_prob_threshold = var_map["replace_prob_threshold"].as<float>();
    
    string model_path;
    if (0 == var_map.count("model"))
    {
        BOOST_LOG_TRIVIAL(fatal) << "model path must be specified ! \n Exit !";
        return -1;
    }
    else model_path = var_map["model"].as<string>();
    ofstream model_os(model_path); // check whether model path is ok 
    if (!model_os)
    {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open model path at '" << model_path << "'. \n Exit !";
        return -1;
    }
    
    // others will be processed flowing 
    
    // Init 
    cnn::Initialize(argc , argv , 1234); // 
    BILSTMModel4Tagging tagging_model;

    // reading traing data , get word dict size and output tag number
    // -> set replace frequency for word_dict_wrapper
    tagging_model.word_dict_wrapper.set_threshold(replace_freq_threshold, replace_prob_threshold);
    ifstream train_is(training_data_path);
    if (!train_is) {
        BOOST_LOG_TRIVIAL(fatal) << "failed to open training: `" << training_data_path << "` .\n Exit! \n";
        return -1;
    }
    vector<InstancePair> training_samples;
    tagging_model.read_training_data_and_build_dicts(train_is, &training_samples);
    tagging_model.finish_read_training_data();
    train_is.close();

    // build model structure
    tagging_model.build_model_structure(var_map); // passing the var_map to specify the model structure

    // reading word embedding
    if (0 != var_map.count("word_embedding"))
    {
        string word_embedding_path = var_map["word_embedding"].as<string>();
        ifstream word_embedding_is(word_embedding_path);
        if (!word_embedding_is)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to open wordEmbedding at : `" << word_embedding_path << "` .\n Exit! \n";
            return -1;
        }
        BOOST_LOG_TRIVIAL(info) << "load word embedding from file `" << word_embedding_path << "`";
        tagging_model.load_wordembedding_from_word2vec_txt_format(word_embedding_is);
        word_embedding_is.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(info) << "No word embedding is specified . Using randomized initialization .";
    }


    // reading developing data
    std::vector<InstancePair> devel_samples, *p_devel_samples;
    if ("" != devel_data_path)
    {
        std::ifstream devel_is(devel_data_path);
        if (!devel_is) {
            BOOST_LOG_TRIVIAL(error) << "failed to open devel file: `" << devel_data_path << "`\n Exit!";
            // if set devel data , but open failed , we exit .
            return -1;
        }
        tagging_model.read_devel_data(devel_is, &devel_samples);
        devel_is.close();
        p_devel_samples = &devel_samples;
    }
    else p_devel_samples = nullptr;

    // Train 
    tagging_model.train(&training_samples, max_epoch, p_devel_samples, devel_freq);

    // save model
    tagging_model.save_model(model_os);
    model_os.close();
    return 0;
}
int devel_process(int argc, char *argv[] , const string &program_name) 
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Validation(develop) process "
        "using `" + program_name + " devel <options>` to validate . devel options are as following";
    po::options_description op_des = po::options_description(description);
    // set params to receive the arguments 
    string devel_data_path , model_path, error_output_path;
    op_des.add_options()
        ("devel_data", po::value<string>(&devel_data_path), "The path to validation data .")
        ("model", po::value<string>(&model_path), "Use to specify the model name(path)")
        ("error_output", po::value<string>(&error_output_path), "Specify the file path to storing the predict error infomation . Empty to discard.")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc , argv).options(op_des).allow_unregistered().run(), var_map);
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
    BILSTMModel4Tagging tagging_model;

    // Load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n"
            "Exit !";
        return -1;
    }
    tagging_model.load_model(is);
    is.close();

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

    vector<InstancePair> devel_samples;
    tagging_model.read_devel_data(devel_is , &devel_samples);
    devel_is.close();

    // devel
    tagging_model.devel(&devel_samples , p_error_output_os); // Get the same result , it is OK .
    if (p_error_output_os)
    {
        p_error_output_os->close();
        delete p_error_output_os;
    }
    return 0;
}

int predict_process(int argc, char *argv[] , const string &program_name) 
{
    string description = PROGRAM_DESCRIPTION + "\n"
        "Predict process ."
        "using `" + program_name + " predict <options>` to predict . predict options are as following";
    po::options_description op_des = po::options_description(description);
    op_des.add_options()
        ("raw_data", po::value<string>(), "The path to raw data(It should be segmented) .")
        ("output" , po::value<string>() , "The path to storing result . using `stdout` if not specified ." )
        ("model", po::value<string>(), "Use to specify the model name(path)")
        ("help,h", "Show help information.");
    po::variables_map var_map;
    po::store(po::command_line_parser(argc , argv).options(op_des).allow_unregistered().run(), var_map);
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
    BILSTMModel4Tagging tagging_model;

    // load model 
    ifstream is(model_path);
    if (!is)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open model path at '" << model_path << "' . \n"
            "Exit .";
        return -1;
    }
    tagging_model.load_model(is);
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
        tagging_model.predict(raw_is, cout); // using `cout` as output stream 
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
        tagging_model.predict(raw_is, os);
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
    else if (string(argv[1]) == "train") return train_process(argc-1, argv+1 , argv[0]);
    else if (string(argv[1]) == "devel") return devel_process(argc-1, argv+1 , argv[0]);
    else if (string(argv[1]) == "predict") return predict_process(argc-1, argv+1 , argv[0]);
    else
    {
        cerr << "unknown mode : " << argv[1] << "\n"
            << usage;
        return -1;
    }
}
