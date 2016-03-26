#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>

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

void read_dataset_and_build_dicts(istream &is, vector<InstancePair> & samples, cnn::Dict & word_dict, cnn::Dict & tag_dict)
{
	BOOST_LOG_TRIVIAL(info) << "reading instance";
	unsigned line_cnt = 0;
	string line;
	vector<InstancePair> tmp_samples;
	vector<int> sent, tag_seq;
	sent.resize(256);
	tag_seq.resize(256);
	while (getline(is, line))
	{
		boost::algorithm::trim(line);
		vector<string> strpair_cont;
		boost::algorithm::split(strpair_cont, line, boost::is_any_of("\t"));
		sent.clear();
		tag_seq.clear();
		for (string &strpair : strpair_cont)
		{
			/*word_and_tag.clear();
			boost::algorithm::split(word_and_tag , strpair , boost::is_any_of("_"));
			assert(2 == word_and_tag.size());
			int word_id = word_dict.Convert(word_and_tag[0]);
			int tag_id = tag_dict.Convert(word_and_tag[1]);*/
			string::size_type  delim_pos = strpair.find_last_of("_");
			assert(delim_pos != string::npos);
			int word_id = word_dict.Convert(strpair.substr(0, delim_pos));
			int tag_id = tag_dict.Convert(strpair.substr(delim_pos + 1));
			sent.push_back(word_id);
			tag_seq.push_back(tag_id);
		}
		tmp_samples.emplace_back(sent, tag_seq); // using `pair` construction pair(first_type , second_type)
		++line_cnt;
		if (0 == line_cnt % 4) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << "lines"; break; }
	}
	tmp_samples.swap(samples);
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
	Index SOS;
	Index EOS;

	struct AccStat
	{
		unsigned long correct_tags;
		unsigned long total_tags;
		AccStat() :correct_tags(0), total_tags(0) {};
		float get_acc() { return total_tags != 0 ? float(correct_tags) / total_tags : 0.; }
		void clear() { correct_tags = 0; total_tags = 0;  }
		AccStat &operator+=(const AccStat &other) { correct_tags += other.correct_tags; total_tags += other.total_tags; return *this; }
		AccStat operator+(const AccStat &other) { AccStat tmp = *this;  tmp += other; return tmp; }
	};

	BILSTMModel4Tagging(const po::variables_map& conf) :
		m(new Model())
	{
		LSTM_LAYER = conf["lstm_layers"].as<unsigned>();
		INPUT_DIM = conf["input_dim"].as<unsigned>();
		LSTM_HIDDEN_DIM = conf["lstm_hidden_dim"].as<unsigned>();
		TAG_HIDDEN_DIM = conf["tag_dim"].as<unsigned>();

		l2r_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);
		r2l_builder = new LSTMBuilder(LSTM_LAYER, INPUT_DIM, LSTM_HIDDEN_DIM, m);
		SOS = word_dict.Convert(SOS_STR);
		EOS = word_dict.Convert(EOS_STR);
	}

	~BILSTMModel4Tagging()
	{
		delete m;
		delete l2r_builder;
		delete r2l_builder;
	}

	void set_model_structure_after_fill_dict()
	{
		TAG_OUTPUT_DIM = tag_dict.size();
		WORD_DICT_SIZE = word_dict.size();
	}

	void freeze_dict()
	{
		tag_dict.Freeze();
		word_dict.Freeze();
	}

	void print_model_info()
	{
		cout << "------------Model structure info-----------\n"
			<< "vocabulary(word dict) size : " << WORD_DICT_SIZE << " with dimension : " << INPUT_DIM << "\n"
			<< "LSTM hidden layer dimension : " << LSTM_HIDDEN_DIM << " , has " << LSTM_LAYER << " layers\n"
			<< "TAG hidden layer dimension : " << TAG_HIDDEN_DIM << "\n"
			<< "output dimention(tags number) : " << TAG_OUTPUT_DIM << "\n"
			<< "--------------------------------------------" << endl;
	}

	void init_model_params()
	{
		if (0 == WORD_DICT_SIZE || 0 == TAG_OUTPUT_DIM) {
			BOOST_LOG_TRIVIAL(error) << "call `set_model_structure_after_fill_dict` to set output dim and word dict size";
			abort();
		}

		words_lookup_param = m->add_lookup_parameters(WORD_DICT_SIZE, { INPUT_DIM });
		l2r_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
		r2l_tag_hidden_w_param = m->add_parameters({ TAG_HIDDEN_DIM , LSTM_HIDDEN_DIM });
		tag_hidden_b_param = m->add_parameters({ TAG_HIDDEN_DIM });

		tag_output_w_param = m->add_parameters({ TAG_OUTPUT_DIM , TAG_HIDDEN_DIM });
		tag_output_b_param = m->add_parameters({ TAG_OUTPUT_DIM });
	}

	void load_wordembedding_from_word2vec_txt_format(const string &fpath)
	{
		// TODO set lookup parameters from outer word embedding
		// using words_loopup_param.Initialize( word_id , value_vector )
	}

	Expression negative_loglikelihood(const IndexSeq *p_sent, const IndexSeq *p_tag_seq, ComputationGraph *p_cg, AccStat *p_stat=nullptr)
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
			word_lookup_exp_cont[i] = noise(word_lookup_exp, 0.1);
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

	Expression predict(const IndexSeq *p_sent, ComputationGraph *p_cg, IndexSeq *p_predict_tag_seq)
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
			Expression word_lookup_exp = lookup(cg, words_lookup_param, p_sent->at(i));
			word_lookup_exp_cont[i] = noise(word_lookup_exp, 0.1);
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

	void train(const vector<InstancePair> *samples, unsigned max_epoch, const vector<InstancePair> *dev_samples = nullptr)
	{
		unsigned nr_samples = samples->size();
		vector<unsigned> access_order(nr_samples);
		for (unsigned i = 0; i < nr_samples; ++i) access_order[i] = i;
		
		SimpleSGDTrainer sgd = SimpleSGDTrainer(m);

		for (unsigned nr_epoch = 0; 0 < max_epoch; ++nr_epoch)
		{
			// For loss , accuracy report and 
			AccStat training_stat_per_report , training_stat_per_epoch;
			unsigned report_freq = 10000;
			double loss_per_report = 0.,
				loss_per_epoch = 0.;
			for (unsigned i = 0; i < nr_samples; ++i)
			{
				const InstancePair &instance_pair = samples->at(access_order[i]);
				// using negative_loglikelihood loss to build model
				const IndexSeq *p_sent = &instance_pair.first,
					*p_tag_seq = &instance_pair.second;
				ComputationGraph cg;
				negative_loglikelihood(p_sent, p_tag_seq, &cg, &training_stat_per_report);
				loss_per_report += as_scalar(cg.forward());
				cg.backward();
				sgd.update(1.0);

				if (0 == i % report_freq) // Report 
				{
					BOOST_LOG_TRIVIAL(info) << i << " instances have been trained , with E = "
						<< loss_per_report / training_stat_per_report.correct_tags 
						<< " , acc = " << training_stat_per_report.get_acc() * 100 << " % .";
					loss_per_epoch += loss_per_report;
					loss_per_report = 0.;
					training_stat_per_epoch += training_stat_per_report;
					training_stat_per_report.clear();
				}
			}
			sgd.status();

			loss_per_epoch += loss_per_report;
			training_stat_per_epoch += training_stat_per_report;
			BOOST_LOG_TRIVIAL(info) << "epoch " << nr_epoch + 1 << " has been finished . for this epoch , E = " << ;
		}
	}
};

void init_command_line_options(int argc, char* argv[], po::variables_map* conf) {
	po::options_description opts("Configuration");
	opts.add_options()
		("training_data", po::value<std::string>(), "The path to the training data.")
		("devel_data", po::value<std::string>(), "The path to the development data.")
		("test_data", po::value<std::string>(), "The path to the test data.")
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
	cnn::Initialize(argc, argv, 1234); // MUST , or no memory is allocated ! Also the the random seed.
	// -
	// BUILD MODEL
	// -
	// declare model 

	po::variables_map conf;
	init_command_line_options(argc, argv, &conf);

	BILSTMModel4Tagging tagging_model(conf);

	// Reading Samples
	ifstream train_is(conf["training_data"].as<std::string>());
	if (!train_is) {
		BOOST_LOG_TRIVIAL(error) << "failed to open: " << conf["training_data"].as<std::string>();
		abort();
	}
	vector<InstancePair> samples;
	cnn::Dict &word_dict = tagging_model.word_dict,
		&tag_dict = tagging_model.tag_dict;
	read_dataset_and_build_dicts(train_is, samples, word_dict, tag_dict);
	train_is.close();

	// set WORD_DICT_SIZE , TAG_OUTPUT_DIM ,  init params after that
	tagging_model.set_model_structure_after_fill_dict();
	tagging_model.init_model_params();
	tagging_model.print_model_info();
	getchar(); // pause ;
	return 0;
}
