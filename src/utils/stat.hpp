#ifndef STAT_HPP_INCLUDED_
#define STAT_HPP_INCLUDED_
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <stdlib.h>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

#include <boost/algorithm/string/trim.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>

#include "cnn/dict.h"
#include "segmentor/cws_module/cws_tagging_system.h"

/*************************************
 * Stat 
 * for statistics , including loss , acc , time .
 * 
 ************************************/
namespace slnn{

struct BasicStat
{
    float loss;
    unsigned long total_tags;
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    bool is_predict ;
    std::chrono::high_resolution_clock::time_point start_time_stat() 
    {
        time_clock_locked = false;
        return time_start = std::chrono::high_resolution_clock::now(); 
    }
    std::chrono::high_resolution_clock::time_point end_time_stat() 
    { 
        time_clock_locked = true;
        return time_end = std::chrono::high_resolution_clock::now(); 
    }
    BasicStat(bool is_predict=false) :loss(0.f) , total_tags(0) , is_predict(is_predict) , time_clock_locked(false){};
    float get_sum_E(){ return loss ; }
    long long get_time_cost_in_seconds()
    {
        if (!time_clock_locked) 
        {
            end_time_stat();
            time_clock_locked = false;
        }
        std::chrono::seconds du = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);
        return du.count();
    }
    float get_speed_as_kilo_tokens_per_sencond()
    {
        return static_cast<float>(static_cast<long double>(total_tags) / 1000. / get_time_cost_in_seconds());
    }
    BasicStat &operator+=(const BasicStat &other)
    {
        loss += other.loss;
        return *this;
    }
    BasicStat operator+(const BasicStat &other) { BasicStat tmp = *this;  tmp += other;  return tmp; }
    std::string get_stat_str(const std::string &info_header)
    {
        std::ostringstream str_os;
        str_os << info_header << "\n" ;
        if( !is_predict ) str_os << "Total E = " << get_sum_E() << "\n" ;
        str_os << "Time cost = " << get_time_cost_in_seconds() << " s\n"
            << "Speed = " << get_speed_as_kilo_tokens_per_sencond() << " K tokens/s";
        return str_os.str();
    }
protected :
    bool time_clock_locked;

};

struct PostagStat : public BasicStat
{
    unsigned long correct_tags ;
    
    PostagStat(bool is_predict=false) : BasicStat(is_predict) , correct_tags(0){};
    float get_acc() { return total_tags != 0 ? float(correct_tags) / total_tags : 0.f; }
    float get_E() { return total_tags != 0 ? loss / total_tags : 0.f; }
    void clear() { correct_tags = 0; total_tags = 0; loss = 0.f; }
    PostagStat &operator+=(const PostagStat &other)
    {
        correct_tags += other.correct_tags;
        total_tags += other.total_tags;
        loss += other.loss;
        return *this;
    }
    PostagStat operator+(const PostagStat &other) { PostagStat tmp = *this;  tmp += other;  return tmp; }
    std::string get_stat_str(const std::string &info_header)
    {
        std::ostringstream str_os;
        str_os << info_header << "\n"
            << "Average E = " << get_E() << "\n"
            << "Acc = " << get_acc() * 100 << "% \n" 
            << "Time cost = " << get_time_cost_in_seconds() << " s\n"
            << "Speed = " << get_speed_as_kilo_tokens_per_sencond() << " K tokens/s\n"
            << "Total tags = " << total_tags << " , Correct tags = " << correct_tags ;
        return str_os.str();
    }
    std::string get_basic_stat_str(const std::string &info_header)
    {
        return BasicStat::get_stat_str(info_header);
    }
};

using Stat = PostagStat;

struct NerStat : BasicStat
{
    std::string eval_script_path;
    std::string tmp_output_path;
    NerStat(const std::string &eval_script_path , const std::string &tmp_output_path=std::string("eval_out.tmp") , 
            bool is_predict=false) :
        BasicStat(is_predict),
        eval_script_path(eval_script_path), tmp_output_path(tmp_output_path)
    {}

    std::array<float , 4>
    conlleval(const std::vector<IndexSeq> &gold_ner_seqs ,
        const std::vector<IndexSeq> &predict_ner_seqs , 
        const cnn::Dict &ner_dict) 
    {
        std::array<float , 4> fake_ret = {100.f , 100.f , 100.f , 100.f } ;
#ifndef _MSC_VER
        std::array<float, 4> ret;
        // write `WORD GOLD_NER PREDICT_NER` to temporal output , where we using fake `WORD` , it is no use for evaluation result
        // - set unique output path
        unsigned timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count() ;
        unsigned pid = static_cast<unsigned>(::getpid()) ;
        tmp_output_path = tmp_output_path + "_" + std::to_string(pid) + "_" + std::to_string(timestamp) ;
        std::ofstream tmp_of(tmp_output_path);
        if (!tmp_of)
        { 
            BOOST_LOG_TRIVIAL(fatal) << "Failed to create temporial output file for evaltion `" << tmp_output_path
                << "`";
            abort();
        }
        for (size_t seq_idx = 0; seq_idx < predict_ner_seqs.size(); ++seq_idx)
        {
            const IndexSeq &ner_seq = gold_ner_seqs.at(seq_idx),
                &predict_seq = predict_ner_seqs.at(seq_idx);
            for (size_t token_idx = 0; token_idx < predict_seq.size(); ++token_idx)
            {
                tmp_of << "W" << " "
                    << ner_dict.Convert(ner_seq.at(token_idx)) << " "
                    << ner_dict.Convert(predict_seq.at(token_idx)) << "\n";
            }
            tmp_of << "\n"; // split of one sequence
        }
        tmp_of.close();
        std::string cmd = eval_script_path + " " + tmp_output_path;
        BOOST_LOG_TRIVIAL(info) << "Running: " << cmd << std::endl;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            return fake_ret ;
        }
        char buffer[128];
        std::string result = "";
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL) { result += buffer; }
        }
        pclose(pipe);
        // rm tmp output file
        pipe = popen( (std::string("rm -f ") + tmp_output_path).c_str() , "r") ;
        if(!pipe)
        {
            std::cerr << "Error . failed to rm temp output file : `" << tmp_output_path << "`\n" ; 
        }
        else
        {
            pclose(pipe) ;
        }
        std::stringstream S(result);
        std::string token;
        unsigned idx = 0 ;
        while (S >> token) {
            boost::algorithm::trim(token);
            ret[idx++] = stof(token);
        }
        return ret ;
#else
        return fake_ret ;
#endif
        return fake_ret ;
    }
};

struct CWSStat : BasicStat
{
    CWSTaggingSystem &tag_sys ;
    CWSStat(CWSTaggingSystem &tag_sys , bool is_predict=false)
      :BasicStat(is_predict) ,
        tag_sys(tag_sys)
    {}

    // return : {Acc , P , R , F1}
    std::array<float , 4>
    eval(const std::vector<IndexSeq> &gold_seqs, const std::vector<IndexSeq> &pred_seqs)
    {
        size_t seq_num = gold_seqs.size() ;
        unsigned gold_tokens = 0,
            found_tokens = 0,
            correct_tokens = 0 ; // P , R , F1
        unsigned total_tags = 0,
            correct_tags = 0 ; // ACC
        for( size_t i = 0 ; i < seq_num ; ++i )
        {
            std::array<unsigned, 3> result = eval_one_seq(gold_seqs[i], pred_seqs[i]) ;
            correct_tokens += result[0] ;
            gold_tokens += result[1] ;
            found_tokens += result[2] ;
            total_tags += gold_seqs.size() ;
            for( size_t pos = 0 ; pos < gold_seqs[i].size() ; ++pos )
            {
                if( gold_seqs[i][pos] == pred_seqs[i][pos] ) ++correct_tags ;
            }
        }
        float Acc = (total_tags == 0) ? 0.f : static_cast<float>(correct_tags) / total_tags ;
        float P = (found_tokens == 0) ? 0.f : static_cast<float>(correct_tokens) / found_tokens ;
        float R = (gold_tokens == 0) ? 0.f : static_cast<float>(correct_tokens) / gold_tokens ;
        float F1 = (std::abs(R + P - 0.f) < 1e-6) ? 0.f : 2 * P * R / (P + R) ;
        return std::array<float, 4>{Acc, P, R, F1} ;
    }

    std::array<unsigned , 3>
    eval_one_seq(const IndexSeq &gold_seq, const IndexSeq &pred_seq)
    {
        std::vector<std::array<unsigned, 2>> gold_words,
            pred_words ;
        parse_tag_seq2word_range(gold_seq, gold_words) ;
        parse_tag_seq2word_range(pred_seq, pred_words) ;
        unsigned gold_word_size = gold_words.size(),
            pred_word_size = pred_words.size() ;
        size_t gold_pos = 0 ,
            pred_pos = 0 ;
        unsigned correct_cnt = 0 ;
        while( gold_pos < gold_word_size && pred_pos < pred_word_size )
        {
            std::array<unsigned, 2> &gold_word = gold_words[gold_pos],
                &pred_word = pred_words[pred_pos] ;
            if( gold_word[0] == pred_word[0] )
            {
                // word is aligned
                if( gold_word[1] == pred_word[1] )
                {
                    ++correct_cnt ;
                }
            }
            ++gold_pos ;
            // try to align
            unsigned gold_char_pos = gold_words[gold_pos][0] ;
            while( pred_pos < pred_word_size && pred_words[pred_pos][0] < gold_char_pos ) 
                ++pred_pos ;
        }
        return std::array<unsigned, 3>{correct_cnt, gold_word_size, pred_word_size} ;
    }
    void parse_tag_seq2word_range(const IndexSeq &seq, std::vector<std::array<unsigned, 2>> &word_ranges)
    {
        std::vector<std::array<unsigned, 2>> tmp_word_ranges ;
        unsigned range_s = 0 ;
        for( unsigned i = 0 ; i < seq.size() ; ++i )
        {
            Index tag_id = seq.at(i) ;
            if( tag_id == tag_sys.S_ID )
            {
                tmp_word_ranges.push_back({ i , i }) ;
                range_s = i + 1 ;
            }
            else if( tag_id == tag_sys.E_ID )
            {
                tmp_word_ranges.push_back({ range_s , i }) ;
                range_s = i + 1 ;
            }
        }
        std::swap(word_ranges, tmp_word_ranges) ;
    }
};

} // End of namespace slnn

#endif
