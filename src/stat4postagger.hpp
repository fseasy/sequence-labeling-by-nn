#include <chrono>
#include <vector>
#include <string>
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
    chrono::high_resolution_clock::time_point time_start;
    chrono::high_resolution_clock::time_point time_end;
    chrono::high_resolution_clock::time_point start_time_stat() { return time_start = chrono::high_resolution_clock::now(); }
    chrono::high_resolution_clock::time_point end_time_stat() { return time_end = chrono::high_resolution_clock::now(); }
    BasicStat() :loss(0.f) {};
    long long get_time_cost_in_seconds()
    {
        chrono::seconds du = chrono::duration_cast<chrono::seconds>(time_end - time_start);
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
        return (long double)(correct_tags) / 1000. / get_time_cost_in_seconds();
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
    string eval_script_path;
    string tmp_output_path;
    NerStat(const string &eval_script_path , const string &tmp_output_path=string("eval_out.tmp")) :BasicStat(),
        eval_script_path(eval_script_path), tmp_output_path(tmp_output_path)
    {

    }

    double conlleval(const std::vector<IndexSeq> gold_ner_seqs ,
        const std::vector<IndexSeq> predict_ner_seqs , const cnn::Dict &ner_dict) {
#ifndef _MSC_VER
        std::string cmd = conf["conlleval"].as<std::string>() + " " + tmp_output;
        _TRACE << "Running: " << cmd << std::endl;
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
            return boost::lexical_cast<double>(token);
}
#else
        return 1.;
#endif
        return 0.;
    }

};

} // End of namespace slnn