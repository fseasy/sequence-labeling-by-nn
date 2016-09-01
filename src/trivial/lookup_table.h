#ifndef SLNN_TRIVIAL_LOOKUP_TABLE_H_
#define SLNN_TRIVIAL_LOOKUP_TABLE_H_
#include <unordered_map>
#include <string>
#include <random>
#include <functional>
namespace slnn{

class LookupTable
{
    friend class boost::serialization::access;
public:
    using Index = int; // we should use the global declaration. But it seems so duplicted.
public:
    LookupTable();
    
    Index convert(const std::string &str);
    Index convert(const std::string &str) const;
    const std::string& convert(Index idx) const;
    const std::string& convert_ban_unk(Index idx) const; // if idx=unk, domain_error exception will be throw
    
    void set_unk();
    Index get_unk_idx() const;
    bool has_set_unk() const;
    bool is_unk_idx() const;
    
    void freeze();
    void unfreeze();
    bool has_frozen() const;

    std::size_t count(const std::string &str) const;
    std::size_t count(Index idx) const;
    std::size_t count_ban_unk(Index idx) const;
    
    void reset();
    std::size_t size() const;
    std::size_t size_without_unk() const;

private:
    std::unordered_map<std::string, Index> str2idx;
    std::vector<std::string> idx2str;
    bool is_unk_seted;
    bool is_frozen;
    static const std::string UnkStr;
    Index unk_idx;

private:
    template<class Archive> void serialize(Archive& ar, const unsigned int) 
    {
        ar &str2idx &idx2str &is_unk_seted &is_frozen &unk_idx;
    }
};

const std::string LookupTable::UnkStr = "*u*n*k*";


class LookupTableWithCnt : public LookupTable
{
    friend class boost::serialization::access;
public:
    // hidden the base class function with the same name
    Index convert(const std::string &str);
    std::size_t count(const std::string &str) const;
    std::size_t count(Index idx) const;
    std::size_t count_ban_unk(Index idx) const;

    void LookupTable::reset();
private:
    std::vector<unsigned> cnt;
};

template<class Archive> 
void LookupTableWithCnt::serialize(Archive& ar, const unsigned int) 
{
    ar &cnt;
}


class LookupTableWithReplace : public LookupTableWithCnt
{
    friend class boost::serialization::access;
public:
    LookupTableWithReplace(std::size_t seed=1234, unsigned cnt_threshold=1, float prob_threshold=0.2f);
    template <typename Generator>
    LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold=1, float prob_threshold=0.2f);
    
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold);

    Index unk_replace_in_probability(Index idx) const; 
private:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::function<float()> rand_generator;
    unsigned cnt_threshold;
    float prob_threshold;
};


template <typename Generator>
LookupTableWithReplace::LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold, float prob_threshold)
    :rand_generator(std::bind(std::normal_real_distribution<float>(0,1), rng)),
    cnt_threshold(cnt_threshold),
    prob_threshold(prob_threshold)
{}

template<class Archive> 
void LookupTableWithReplace::serialize(Archive& ar, const unsigned int)
{
    ar &cnt_threshold &prob_threshold;
}

} // end of namespace slnn

#endif