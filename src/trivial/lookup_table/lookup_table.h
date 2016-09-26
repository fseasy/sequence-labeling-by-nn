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

template<typename TokenType>
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
    Index convert(const TokenType &str); 
    
    /* convert str to index
        Exception : out_of_range 
        Return : 1. corresponding index, if: str in dict
                 2. unk_idx, if: str not in dict && has_set_unk
                 3. un-excepted
    */
    Index convert(const TokenType &str) const; // convert str to idx(never add(no ))
    
    /* convert index to str (ban unk_index)
        Exception: 1. domain_error, if use unk_idx as parameter when has_set_unk
                   2. out_of_range, index not in range
        Return: 1. corresponding index, if : index is valid
                2. un-excepted(exception throw)
    */
    const TokenType& convert_ban_unk(Index idx) const; // if idx=unk, domain_error exception will be throw
    
    void set_unk();

    /* get unk index
        Exception: logic_error, if: !has_set_unk
        Return: 1. unk_idx, if: has_set_unk
                2. un-excepted(exception throw)
    */
    Index get_unk_idx() const;
    bool has_set_unk() const noexcept;
    bool is_unk_idx(Index idx) const noexcept;
    
    void freeze() noexcept;
    bool has_frozen() const noexcept;

    /* count the str occur times in dict ( 1/0)
        Return: 1. 1, if str in dict
                2. 0, if not
    */
    std::size_t count(const TokenType &str) const noexcept;
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
    /* dirived class may need query token2idx directly */
    const std::unordered_map<TokenType, Index>& get_token2idx_dict() const noexcept{ return token2idx; }

protected:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::unordered_map<TokenType, Index> token2idx;
    std::vector<TokenType> idx2token;
    bool is_unk_seted;
    bool is_frozen;
    static constexpr Index UnkUnsetValue = -1;
    Index unk_idx;
};



template <typename TokenType>
class LookupTableWithCnt : public LookupTable<TokenType>
{
    friend class boost::serialization::access;
public:
    // hidden the base class function with the same name
    Index convert(const TokenType &str);
    using LookupTable::convert; // look at C++ Primer (Chinese Version) P551. to make other `convert` is visitable

    /* count str occurrence times (N)
        Return : N, occurrence times
    */
    std::size_t count(const TokenType &str) const noexcept;
    std::size_t count_ban_unk(Index idx) const;

    void reset() noexcept;

protected:
    template<class Archive> 
    void serialize(Archive& ar, const unsigned int);

private:
    std::vector<unsigned> cnt;
};


template<typename TokenType>
class LookupTableWithReplace : public LookupTableWithCnt<TokenType>
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


/*************************************
 *     LookupTable Implementation
 *************************************/

template <typename TokenType>
LookupTable<TokenType>::LookupTable() noexcept
    :is_unk_seted(false),
    is_frozen(false),
    unk_idx(UnkUnsetValue)
{}

template <typename TokenType>
typename LookupTable<TokenType>::Index 
LookupTable<TokenType>::convert(const TokenType &token)
{
    auto iter = token2idx.find(token);
    if( iter != token2idx.cend() )
    { 
        return iter->second; 
    } 
    else
    {
        // not find
        if( has_frozen() )
        { 
            if( has_set_unk() ){ return unk_idx; }
            else { throw std::out_of_range("key '" + token + "' was not in LookupTable."); }
        }
        else
        {
            // add new key
            Index nextIndex = token2idx.size();
            token2idx[token] = nextIndex; // add str -> idx
            idx2token.push_back(token); // add idx -> str
            return nextIndex;
        }
    }
}

template <typename TokenType>
typename LookupTable<TokenType>::Index 
LookupTable<TokenType>::convert(const TokenType &token) const
{
    // only read, no write(add)
    auto iter = token2idx.find(token);
    if( iter != token2idx.cend() ){ return iter->second; }
    else 
    { 
        if( has_set_unk() ){ return unk_idx; }
        else { throw std::out_of_range("key '" + token + "' was not in LookupTable.") ; }
    }
}

template <typename TokenType>
const TokenType&
LookupTable<TokenType>::convert_ban_unk(Index idx) const
{
    if( idx == unk_idx && has_set_unk() )
    { 
        if( has_set_unk() ){ throw domain_error("unk index('" + to_string(idx) + "') was banned."); }
        else
        {
            throw std::out_of_range("index '" + to_string(idx) + "' was out of range( size = " +
                to_string(size()) + ")");
        }
    }
    else
    {
        if( idx >= static_cast<Index>(size()) || idx < 0 )
        { 
            throw std::out_of_range("index '" + to_string(idx) + "' was out of range( size = " +
                to_string(size()) + ")");
        }
        else{ return idx2token[idx]; }
    }
}

template<typename TokenType>
void LookupTable<TokenType>::set_unk()
{
    if( has_set_unk() ){ return; }
    if( ! has_frozen() )
    {
        throw std::logic_error("before set unk, lookup table should be frozen firstly.")
    }
    else{ unk_idx = token2idx.size(); }
}

template<typename TokenType>
typename LookupTable<TokenType>::Index 
LookupTable<TokenType>::get_unk_idx() const
{
    if( !has_set_unk() ){ throw std::logic_error("unk was not set."); }
    else{ return unk_idx; }
}

template <typename TokenType>
bool LookupTable<TokenType>::has_set_unk() const noexcept
{
    return unk_idx != UnkUnsetValue;
}

template <typename TokenType>
bool LookupTable<TokenType>::is_unk_idx(Index idx) const noexcept 
{
    if( !has_set_unk() ){ return false; }
    else{ return idx == unk_idx; }
}

template <typename TokenType>
void LookupTable<TokenType>::freeze() noexcept
{
    is_frozen = true;
}

template <typename TokenType>
bool LookupTable<TokenType>::has_frozen() const noexcept
{
    return is_frozen;
}

template <typename TokenType>
size_t LookupTable<TokenType>::count(const TokenType& token) const noexcept 
{
    return token2idx.find(token) != token2idx.end();
}

template <typename TokenType>
size_t LookupTable<TokenType>::count_ban_unk(Index idx) const
{
    if( idx == unk_idx && has_set_unk() ){ throw std::domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else 
    { 
        return idx >= 0 && idx < size_without_unk(); 
    }
}

template <typename TokenType>
void LookupTable<TokenType>::reset() noexcept
{
    token2idx.clear();
    idx2token.clear();
    is_frozen = false;
    unk_idx = UnkUnsetValue;
}

template <typename TokenType>
size_t LookupTable<TokenType>::size() const noexcept 
{
    if( has_set_unk() ){ return token2idx.size() + 1U; }
    else { return token2idx.size();  }
}

template <typename TokenType>
size_t LookupTable<TokenType>::size_without_unk() const noexcept
{
    return token2idx.size();
}

template <typename TokenType>
template<class Archive> 
void LookupTable<TokenType>::serialize(Archive& ar, const unsigned int) 
{
    ar &token2idx &idx2token &is_frozen &unk_idx;
}

/************************************
 *   Implementation  LookupTableWithCnt
 ************************************/
/*------------ LookupTable with Count ------------------*/

template <typename TokenType>
typename LookupTableWithCnt<TokenType>::Index
LookupTableWithCnt<TokenType>::convert(const TokenType& token)
{
    Index idx = LookupTable::convert(token);
    if( !has_frozen() )
    {
        if( idx == static_cast<int>(cnt.size()) )
        {
            // has added one!
            cnt.push_back(1);
        }
        else{ ++cnt[idx]; }
    }
    return idx;
}

template <typename TokenType>
std::size_t LookupTableWithCnt<TokenType>::count(const TokenType &token) const noexcept 
{
    auto iter = get_str2idx_dict().find(token);
    if( iter == get_str2idx_dict().cend() ){ return 0U; }
    else{ return cnt[iter->second]; }
}

template <typename TokenType>
std::size_t LookupTableWithCnt<TokenType>::count_ban_unk(Index idx) const
{
    if( idx == get_unk_idx()  && has_set_unk() ){ throw std::domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else
    { 
        if( idx >= 0 && idx < size_without_unk() ){ return cnt[idx]; }
        else{ return 0U; }
    }
}

template <typename TokenType>
void LookupTableWithCnt<TokenType>::reset() noexcept
{
    LookupTable::reset();
    cnt.clear();
}

template <typename TokenType>
template<class Archive> 
void LookupTableWithCnt<TokenType>::serialize(Archive& ar, const unsigned int version) 
{
    ar &cnt;
    LookupTable::serialize(ar, version);
}

/***************************************
 *   Implementation LookupTable with Replace
 ***************************************/
template <typename TokenType>
template <typename Generator>
LookupTableWithReplace<TokenType>::LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold, float prob_threshold) noexcept
    :rand_generator(std::bind(std::uniform_real_distribution<float>(0,1), rng)),
    cnt_threshold(cnt_threshold),
    prob_threshold(prob_threshold)
{}

template <typename TokenType>
LookupTableWithReplace<TokenType>::LookupTableWithReplace(std::size_t seed, unsigned cnt_threshold, float prob_threshold) noexcept
    :LookupTableWithReplace(std::mt19937(seed), cnt_threshold, prob_threshold)
{}

template <typename TokenType>
void LookupTableWithReplace<TokenType>::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept
{
    this->cnt_threshold = cnt_threshold;
    this->prob_threshold = prob_threshold;
}

template <typename TokenType>
typename LookupTableWithReplace<TokenType>::Index
LookupTableWithReplace<TokenType>::unk_replace_in_probability(Index idx) const
{
    if( !has_set_unk() ){ throw std::logic_error("unk was not set."); }
    else if( idx == get_unk_idx() ){ return idx; }
    else if( idx >= 0 && idx < static_cast<Index>(size_without_unk()) )
    {
        if( count_ban_unk(idx) <= cnt_threshold && rand_generator() <= prob_threshold ){ return get_unk_idx(); }
        else{ return idx; }
    }
    else
    {
        throw std::out_of_range("index '" + to_string(idx) +
            "' was out of range( size = " + to_string(size()) + ")");
    }
}

template <typename TokenType>
template<class Archive> 
void LookupTableWithReplace<TokenType>::serialize(Archive& ar, const unsigned int version)
{
    ar &cnt_threshold &prob_threshold;
    LookupTableWithCnt::serialize(ar, version);
}

} // end of namespace slnn

#endif