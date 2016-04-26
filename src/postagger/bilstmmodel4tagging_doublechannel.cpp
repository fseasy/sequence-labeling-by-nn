#include <fstream>
#include <vector>
#include <string>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "bilstmmodel4tagging_doublechannel.h"

using namespace std;
using namespace cnn;
namespace slnn{

DoubleChannelModel4POSTAG::DoubleChannelModel4POSTAG()
    : m(nullptr),
    merge_doublechannel_layer(nullptr) ,
    bilstm_layer(nullptr),
    merge_bilstm_and_pretag_layer(nullptr),
    tag_output_linear_layer(nullptr),
    dynamic_dict_wrapper(dynamic_dict)
{}

DoubleChannelModel4POSTAG::~DoubleChannelModel4POSTAG()
{
    if (m) delete m;
    if (merge_doublechannel_layer) delete merge_doublechannel_layer;
    if (bilstm_layer) delete bilstm_layer;
    if (merge_bilstm_and_pretag_layer) delete merge_bilstm_and_pretag_layer;
    if (tag_output_linear_layer) delete tag_output_linear_layer;
}

void DoubleChannelModel4POSTAG::freeze_dict_and_add_UNK()
{
    if (dynamic_dict.is_frozen() || fixed_dict.is_frozen() || postag_dict.is_frozen()) return;
    dynamic_dict.Freeze();
    fixed_dict.Freeze();
    postag_dict.Freeze();

    dynamic_dict.SetUnk(UNK_STR);
    fixed_dict.SetUnk(UNK_STR);
}

void DoubleChannelModel4POSTAG::set_partial_model_structure_param_from_outer(boost::program_options::variables_map &varmap)
{
    dynamic_embedding_dim = varmap["dynamic_embedding_dim"].as<unsigned>();
    postag_embedding_dim = varmap["postag_embedding_dim"].as<unsigned>();
    nr_lstm_stacked_layer = varmap["nr_lstm_stacked_layer"].as<unsigned>();
    lstm_x_dim = varmap["lstm_x_dim"].as<unsigned>();
    lstm_h_dim = varmap["lstm_h_dim"].as<unsigned>();
    tag_layer_hidden_dim = varmap["tag_layer_hidden_dim"].as<unsigned>();

    string word2vec_embedding_path = varmap["word2vec_embedding_path"].as<string>();
    ifstream fis(word2vec_embedding_path); // do not check the open status , should be check outer
    string headerline;
    getline(fis , headerline);
    boost::trim_right(headerline);
    vector<string> split_cont;
    boost::split(split_cont, headerline, boost::is_any_of(" "));
    assert(2U == split_cont.size());
    fixed_embedding_dict_size = stol(split_cont[0]);
    fixed_embedding_dim = stol(split_cont[1]);
    // 8  parameters has been inited
}



void DoubleChannelModel4POSTAG::set_partial_model_structure_param_from_inner()
{
    dynamic_embedding_dict_size = dynamic_dict.size();
    tag_layer_output_dim = postag_dict.size();
    assert(fixed_dict.size() == fixed_embedding_dict_size);
}

void DoubleChannelModel4POSTAG::build_model_structure()
{
    assert(dynamic_dict.is_frozen() && fixed_dict.is_frozen() && postag_dict.is_frozen()); // Assert all frozen
    m = new Model();
    merge_doublechannel_layer = new Merge2Layer(m, dynamic_embedding_dim, fixed_embedding_dim, lstm_x_dim);
    bilstm_layer = new BILSTMLayer(m, nr_lstm_stacked_layer, lstm_x_dim, lstm_h_dim);
    merge_bilstm_and_pretag_layer = new Merge3Layer(m, lstm_h_dim, lstm_h_dim, postag_embedding_dim, tag_layer_hidden_dim);
    tag_output_linear_layer = new DenseLayer(m, tag_layer_hidden_dim, tag_layer_output_dim);

    dynamic_words_lookup_param = m->add_lookup_parameters(dynamic_embedding_dict_size, { dynamic_embedding_dim });
    fixed_words_lookup_param = m->add_lookup_parameters(fixed_embedding_dict_size, { fixed_embedding_dim });
    postags_lookup_param = m->add_lookup_parameters(tag_layer_output_dim, { postag_embedding_dim });

    TAG_SOS_param = m->add_parameters({ postag_embedding_dim });

}

void DoubleChannelModel4POSTAG::print_model_info()
{
    BOOST_LOG_TRIVIAL(info) << "---------------- Model Structure Info ----------------\n"
        << "dynamic vocabulary size : " << dynamic_embedding_dict_size << " with dimention : " << dynamic_embedding_dim << "\n"
        << "fixed vocabulary size : " << fixed_embedding_dict_size << " with dimension : " << fixed_embedding_dim << "\n"
        << "bilstm stacked layer num : " << nr_lstm_stacked_layer << " , x dim  : " << lstm_x_dim << " , h dim : " << lstm_h_dim << "\n"
        << "postag num : " << tag_layer_output_dim << " with dimension : " << postag_embedding_dim << "\n"
        << "tag layer hidden dimension : " << tag_layer_hidden_dim << "\n"
        << "tag layer output dimension : " << tag_layer_output_dim;
}

void DoubleChannelModel4POSTAG::save_model(std::ostream &os , stringstream *best_model_tmp_ss)
{
    boost::archive::text_oarchive to(os);
    to << dynamic_embedding_dim << postag_embedding_dim
       << nr_lstm_stacked_layer << lstm_x_dim << lstm_h_dim
       << tag_layer_hidden_dim << fixed_embedding_dim
       << fixed_embedding_dict_size << dynamic_embedding_dict_size
       << tag_layer_output_dim;

    to << dynamic_dict << fixed_dict << postag_dict;
    if (best_model_tmp_ss && 0 != best_model_tmp_ss->rdbuf()->in_avail())
    {
        boost::archive::text_iarchive ti(*best_model_tmp_ss);
        ti >> *m;
        ; // if best model is not save to the temporary stringstream , we should firstly save it !
    }

    to << *m;
    BOOST_LOG_TRIVIAL(info) << "save model done .";
}

void DoubleChannelModel4POSTAG::load_model(std::istream &is)
{
    boost::archive::text_iarchive ti(is);
    ti >> dynamic_embedding_dim >> postag_embedding_dim
        >> nr_lstm_stacked_layer >> lstm_x_dim >> lstm_h_dim
        >> tag_layer_hidden_dim >> fixed_embedding_dim
        >> fixed_embedding_dict_size >> dynamic_embedding_dict_size
        >> tag_layer_output_dim;
    
    ti >> dynamic_dict >> fixed_dict >> postag_dict;
    assert(dynamic_embedding_dict_size == dynamic_dict.size() && fixed_embedding_dict_size == fixed_dict.size()
        && tag_layer_output_dim == postag_dict.size());
    build_model_structure();
    ti >> *m;
    BOOST_LOG_TRIVIAL(info) << "load model done .";
}


Expression DoubleChannelModel4POSTAG::negative_loglikelihood(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, const IndexSeq *p_tag_seq,
    Stat *p_stat = nullptr)
{
    const unsigned sent_len = p_dynamic_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence
    merge_doublechannel_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    merge_bilstm_and_pretag_layer->new_graph(cg);
    tag_output_linear_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> err_exp_cont(sent_len); // for storing every error expression in each tag prediction
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left 
    vector<Expression> pretag_lookup_exp_cont(sent_len); // ADD for PRE_TAG

    // 1. build input , and merge
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression dynamic_word_lookup_exp = lookup(cg, dynamic_words_lookup_param, p_dynamic_sent->at(i));
        Expression fixed_word_lookup_exp = const_lookup(cg, fixed_words_lookup_param, p_fixed_sent->at(i)); // const look up
        dynamic_word_lookup_exp = noise(dynamic_word_lookup_exp, 0.1f);
        fixed_word_lookup_exp = noise(fixed_word_lookup_exp, 0.1f);
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }

    // 2. Build bi-lstm
    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);

    // 3. prepare for PRE_TAG embedding
    Expression TAG_SOS_exp = parameter(cg, TAG_SOS_param);
    pretag_lookup_exp_cont[0] = TAG_SOS_exp;
    for (unsigned i = 1; i < sent_len; ++i)
    {
        pretag_lookup_exp_cont[i] = lookup(cg,postags_lookup_param, p_tag_seq->at(i - 1));
    }

    // build tag network , calc loss Expression of every timestep 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        // rectify is suggested as activation function
        Expression merge_bilstm_pretag_exp = merge_bilstm_and_pretag_layer->build_graph(l2r_lstm_output_exp_cont[i],
            r2l_lstm_output_exp_cont[i], pretag_lookup_exp_cont[i]);
        Expression tag_hidden_layer_output_at_timestep_t = rectify(merge_bilstm_pretag_exp); 
        Expression tag_output_layer_output_at_timestep_t = tag_output_linear_layer->build_graph(tag_hidden_layer_output_at_timestep_t);
        // if statistic , calc output at timestep t
        if (p_stat != nullptr)
        {
            vector<float> output_values = as_vector(cg.incremental_forward());
            Index tag_id_with_max_value = distance(output_values.begin(), max_element(output_values.begin(), output_values.end()));
            ++(p_stat->total_tags); // == ++stat->total_tags ;
            if (tag_id_with_max_value == p_tag_seq->at(i)) ++(p_stat->correct_tags);
        }
        err_exp_cont[i] = pickneglogsoftmax(tag_output_layer_output_at_timestep_t, p_tag_seq->at(i));
    }

    // build the finally loss 
    return sum(err_exp_cont); // in fact , no need to return . just to avoid a warning .
}

void DoubleChannelModel4POSTAG::do_predict(ComputationGraph *p_cg, 
    const IndexSeq *p_dynamic_sent, const IndexSeq *p_fixed_sent, IndexSeq *p_predict_tag_seq)
{
    // The main structure is just a copy from build_bilstm4tagging_graph2train! 
    const unsigned sent_len = p_dynamic_sent->size();
    ComputationGraph &cg = *p_cg;
    // New graph , ready for new sentence 
    // New graph , ready for new sentence
    merge_doublechannel_layer->new_graph(cg);
    bilstm_layer->new_graph(cg);
    merge_bilstm_and_pretag_layer->new_graph(cg);
    tag_output_linear_layer->new_graph(cg);

    bilstm_layer->start_new_sequence();

    // Some container
    vector<Expression> merge_dc_exp_cont(sent_len);
    vector<Expression> l2r_lstm_output_exp_cont; // for storing left to right lstm output(deepest hidden layer) expression for every timestep
    vector<Expression> r2l_lstm_output_exp_cont; // right to left                                                  
    
    // 1. get word embeddings for sent 
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression dynamic_word_lookup_exp = lookup(cg, dynamic_words_lookup_param, p_dynamic_sent->at(i));
        Expression fixed_word_lookup_exp = const_lookup(cg, fixed_words_lookup_param, p_fixed_sent->at(i)); // const look up
        Expression merge_dc_exp = merge_doublechannel_layer->build_graph(dynamic_word_lookup_exp, fixed_word_lookup_exp);
        merge_dc_exp_cont[i] = rectify(merge_dc_exp); // rectify for merged expression
    }
    // 2. calc Expression of every timestep of BI-LSTM

    bilstm_layer->build_graph(merge_dc_exp_cont, l2r_lstm_output_exp_cont, r2l_lstm_output_exp_cont);
    // 3. set previous tag lookup expression

    Expression pretag_lookup_exp = parameter(cg, TAG_SOS_param);

    // build tag network , calc loss Expression of every timestep 
    IndexSeq tmp_predict_tag_seq(sent_len);
    for (unsigned i = 0; i < sent_len; ++i)
    {
        Expression merge_bilstm_pretag_exp = merge_bilstm_and_pretag_layer->build_graph(l2r_lstm_output_exp_cont[i],
            r2l_lstm_output_exp_cont[i], pretag_lookup_exp);
        Expression tag_hidden_layer_output_at_timestep_t = rectify(merge_bilstm_pretag_exp);
        tag_output_linear_layer->build_graph(tag_hidden_layer_output_at_timestep_t);
        vector<float> output_values = as_vector(cg.incremental_forward());
        unsigned tag_id_with_max_value = distance(output_values.cbegin(), max_element(output_values.cbegin(), output_values.cend()));
        tmp_predict_tag_seq.at(i) = tag_id_with_max_value ;
        // set pretag_lookup_exp for next timestep 
        pretag_lookup_exp = lookup(cg, postags_lookup_param, tag_id_with_max_value);
    }
    swap(tmp_predict_tag_seq, *p_predict_tag_seq);
}


} // end of namespace
