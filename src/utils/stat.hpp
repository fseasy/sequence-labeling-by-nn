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
    BasicStat() :loss(0.f) , total_tags(0) , time_clock_locked(false){};
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
        str_os << info_header << "\n"
            << "Total E = " << get_sum_E() << "\n"
            << "Time cost = " << get_time_cost_in_seconds() << " s\n"
            << "Speed = " << get_speed_as_kilo_tokens_per_sencond() << " K tokens/s";
        return str_os.str();
    }
protected :
    bool time_clock_locked;

};

struct PostagStat : public BasicStat
{
    unsigned long correct_tags ;
    
    PostagStat() : BasicStat() , correct_tags(0){};
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
    NerStat(const std::string &eval_script_path , const std::string &tmp_output_path=std::string("eval_out.tmp")) :BasicStat(),
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

} // End of namespace slnn

#endif