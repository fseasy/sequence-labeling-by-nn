#include <chrono>

/*************************************
 * Stat 
 * for statistics , including loss , acc , time .
 * 
 ************************************/
namespace slnn{
struct Stat
{
    unsigned long correct_tags;
    unsigned long total_tags;
    float loss;
    chrono::high_resolution_clock::time_point time_start;
    chrono::high_resolution_clock::time_point time_end;
    Stat() :correct_tags(0), total_tags(0), loss(0) {};
    float get_acc() { return total_tags != 0 ? float(correct_tags) / total_tags : 0.f; }
    float get_E() { return total_tags != 0 ? loss / total_tags : 0.f; }
    void clear() { correct_tags = 0; total_tags = 0; loss = 0.f; }
    chrono::high_resolution_clock::time_point start_time_stat() { return time_start = chrono::high_resolution_clock::now(); }
    chrono::high_resolution_clock::time_point end_time_stat() { return time_end = chrono::high_resolution_clock::now(); }
    long long get_time_cost_in_seconds() 
    { 
        chrono::seconds du = chrono::duration_cast<chrono::seconds>(time_end - time_start);
        return du.count(); 
    }
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
} // End of namespace slnn