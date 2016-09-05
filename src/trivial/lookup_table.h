#ifndef SLNN_TRIVIAL_LOOKUP_TABLE_H_
#define SLNN_TRIVIAL_LOOKUP_TABLE_H_
#include <unordered_map>
#include <string>
#include <random>
#include <functional>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
namespace slnn{

class LookupTable
{
    friend class boost::serialization::access;
public:
    using Index = int; // we should use the global declaration. But it seems so duplicted.
public:
    LookupTable() noexcept;
    
    /* convert str to index 
        if not frozen, add the str to dict when the str was not in dict, and dict size plus 1
        Exception : out_of_range exception will be throw if ( frozen && str not in dict )
        Return : 1. corresponding index if : not frozen
                 2. unk_idx, if : frozen && str not in dict && has_set_unk
                 3. un-excepted(exception will be throw), if: the other condition
    */
    Index convert(const std::string &str); 
    
    /* convert str to index
        Exception : out_of_range 
        Return : 1. corresponding index, if: str in dict
                 2. unk_idx, if: str not in dict && has_set_unk
                 3. un-excepted
    */
    Index convert(const std::string &str) const; // convert str to idx(never add(no ))
    
    /* convert index to str
        Exception : out_of_range, when (index != unk_idx && (idx < 0 || idx >= size
        Return: 1. corresponding index, if: index is valid
                2. un-excepted(exception throw)
    */
    const std::string& convert(Index idx) const;

    /* convert index to str (ban unk_index)
        Exception: 1. domain_error, if use unk_idx as parameter when has_set_unk
                   2. out_of_range, index not in range
        Return: 1. corresponding index, if : index is valid
                2. un-excepted(exception throw)
    */
    const std::string& convert_ban_unk(Index idx) const; // if idx=unk, domain_error exception will be throw
    
    void set_unk() noexcept;

    /* get unk index
        Exception: logic_error, if: !has_set_unk
        Return: 1. unk_idx, if: has_set_unk
                2. un-excepted(exception throw)
    */
    Index get_unk_idx() const;
    bool has_set_unk() const noexcept;
    bool is_unk_idx(Index idx) const noexcept;
    
    void freeze() noexcept;
    void unfreeze() noexcept;
    bool has_frozen() const noexcept;

    /* count the str occur times in dict ( 1/0)
        Return: 1. 1, if str in dict
                2. 0, if not
    */
    std::size_t count(const std::string &str) const noexcept;
    std::size_t count(Index idx) const noexcept;
    
    /* count with ban unk_idx
        Exception: domain_error
        Return: 1. 1
                2. 0
                3. un-excepted
    */
    std::size_t count_ban_unk(Index idx) const;
    
    void reset() noexcept;
    std::size_t size() const noexcept;
    std::size_t size_without_unk() const noexcept;

protected:
    /* dirived class may need query str2idx directly */
    const std::unordered_map<std::string, Index>& get_str2idx_dict() const noexcept{ return str2idx; }

protected:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::unordered_map<std::string, Index> str2idx;
    std::vector<std::string> idx2str;
    bool is_unk_seted;
    bool is_frozen;
    static const std::string UnkStr;
    static constexpr Index UnkUnsetValue = -1;
    Index unk_idx;
};

template<class Archive> 
void LookupTable::serialize(Archive& ar, const unsigned int) 
{
    ar &str2idx &idx2str &is_frozen &unk_idx;
}

class LookupTableWithCnt : public LookupTable
{
    friend class boost::serialization::access;
public:
    // hidden the base class function with the same name
    Index convert(const std::string &str);
    
    // because set_unk may add an key, so we should add cnt if necessary
    void set_unk() noexcept;

    /* count str occurrence times (N)
        Return : N, occurrence times
    */
    std::size_t count(const std::string &str) const noexcept;
    std::size_t count(Index idx) const noexcept;
    std::size_t count_ban_unk(Index idx) const;

    void reset() noexcept;

protected:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::vector<unsigned> cnt;
};

template<class Archive> 
void LookupTableWithCnt::serialize(Archive& ar, const unsigned int version) 
{
    ar &cnt;
    LookupTable::serialize(ar, version);
}


class LookupTableWithReplace : public LookupTableWithCnt
{
    friend class boost::serialization::access;
public:
    LookupTableWithReplace(std::size_t seed = 1234, unsigned cnt_threshold = 1, float prob_threshold = 0.2f) noexcept;
    template <typename Generator>
    LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold = 1, float prob_threshold = 0.2f) noexcept;

    int get_cnt_threshold(){ return cnt_threshold; }
    float get_prob_threshold(){ return prob_threshold; }
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

    Index unk_replace_in_probability(Index idx) const; 

protected:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::function<float()> rand_generator;
    unsigned cnt_threshold;
    float prob_threshold;
};


template <typename Generator>
LookupTableWithReplace::LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold, float prob_threshold) noexcept
    :rand_generator(std::bind(std::uniform_real_distribution<float>(0,1), rng)),
    cnt_threshold(cnt_threshold),
    prob_threshold(prob_threshold)
{}

template<class Archive> 
void LookupTableWithReplace::serialize(Archive& ar, const unsigned int version)
{
    ar &cnt_threshold &prob_threshold;
    LookupTableWithCnt::serialize(ar, version);
}

} // end of namespace slnn

#endif