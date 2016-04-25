#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>

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
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::high_resolution_clock::time_point start_time_stat() { return time_start = std::chrono::high_resolution_clock::now(); }
    std::chrono::high_resolution_clock::time_point end_time_stat() { return time_end = std::chrono::high_resolution_clock::now(); }
    BasicStat() :loss(0.f) {};
    long long get_time_cost_in_seconds()
    {
        std::chrono::seconds du = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);
        return du.count();
    }
    BasicStat &operator+=(const BasicStat &other)
    {
        loss += other.loss;
        return *this;
    }
    BasicStat operator+(const BasicStat &other) { BasicStat tmp = *this;  tmp += other;  return tmp; }
};

struct Stat : public BasicStat
{
    unsigned long correct_tags;
    unsigned long total_tags;
    
    Stat() : BasicStat() , correct_tags(0), total_tags(0) {};
    float get_acc() { return total_tags != 0 ? float(correct_tags) / total_tags : 0.f; }
    float get_E() { return total_tags != 0 ? loss / total_tags : 0.f; }
    void clear() { correct_tags = 0; total_tags = 0; loss = 0.f; }
    float get_speed_as_kilo_tokens_per_sencond()
    {
        return static_cast<float>( (long double)(correct_tags) / 1000. / get_time_cost_in_seconds() );
    }
    Stat &operator+=(const Stat &other)
    {
        correct_tags += other.correct_tags;
        total_tags += other.total_tags;
        loss += other.loss;
        return *this;
    }
    Stat operator+(const Stat &other) { Stat tmp = *this;  tmp += other;  return tmp; }
};

struct NerStat : BasicStat
{
    std::string eval_script_path;
    std::string tmp_output_path;
    NerStat(const std::string &eval_script_path , const std::string &tmp_output_path=std::string("eval_out.tmp")) :BasicStat(),
        eval_script_path(eval_script_path), tmp_output_path(tmp_output_path)
    {

    }

    float conlleval(const std::vector<IndexSeq> gold_ner_seqs ,
        const std::vector<IndexSeq> predict_ner_seqs , const cnn::Dict &ner_dict) {
#ifndef _MSC_VER
        // write `WORD GOLD_NER PREDICT_NER` to temporal output , where we using fake `WORD` , it is no use for evaluation result
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
        BOOST_LOG_TRIVIAL(trace) << "Running: " << cmd << std::endl;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            return 0.;
        }
        char buffer[128];
        std::string result = "";
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL) { result += buffer; }
        }
        pclose(pipe);

        std::stringstream S(result);
        std::string token;
        while (S >> token) {
            boost::algorithm::trim(token);
            return boost::lexical_cast<float>(token);
        }
#else
        return 1.f ;
#endif
        return 0.f ;
    }

};

} // End of namespace slnn
