/**
 * LookupTable, LookupTableWithCnt, LookupTableWithReplace.
 * The kernel function is mapping text-token to integer-index. Using unordered_map to support it.
 * besides, the mainly interface and operation sequence is copying from [c++ neural network - dict.hpp].
 */

#ifndef SLNN_TRIVIAL_LOOKUP_TABLE_H_
#define SLNN_TRIVIAL_LOOKUP_TABLE_H_
#include <unordered_map>
#include <string>
#include <random>
#include <functional>
#include <functional>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
namespace slnn{
namespace trivial{
namespace lookup_table{

/**
 * inner namespace for inner functions.
 */
namespace inner{

/**
 * below 3 overloaded function(including the template) is to satisfy the default demand for token to string.
 */
inline
std::string token2str(const int &token){ return std::to_string(token); };
inline
std::string token2str(const std::string &token){ return token; }
template<typename TokenType>
inline
std::string token2str(const TokenType &token){ return "un-specified token2str"; }

} // end of inner

/**
 * LookupTable, basically support no-count, no-replacement lookup handle.
 * Including add to dict (convert - not frozen), token to index(convert - const or frozen), 
 * index to token (convert - except unk-index), check token/index whethere in dict ,
 * get dict size and so on.
 */

template<typename TokenType, typename Hash=std::hash<TokenType>, typename KeyEqual=std::equal_to<TokenType> >
class LookupTable
{
    friend class boost::serialization::access;
public:
    using Index = int; // we should use the global declaration. But it seems so duplicted.
public:
    LookupTable(
        std::function<std::string(const TokenType&)> token2str_func
        =static_cast<std::string(*)(const TokenType&)>(&inner::token2str)) noexcept;

    /** 
     *  convert str to index(not const), if not frozen, will add the token to dict if firstly occuring.
     *  if not frozen, add the str to dict when the str was not in dict, and dict size plus 1
     *  @param token const reference to TokenType 
     *  @exception  out_of_range exception will be throw if ( frozen && str not in dict )
     *  @return   1. corresponding index if : not frozen
     *            2. unk_idx, if : frozen && str not in dict && has_set_unk
     *            3. un-excepted(exception will be throw), if: the other condition
     *  
     */
    Index convert(const TokenType &token);

    /**
     * convert str to index(const).
     * text token to integer index
     * @param token text token
     * @exception  out_of_range
     * @return   1. corresponding index, if: str in dict
     *           2. unk_idx, if: str not in dict && has_set_unk
     *           3. un-excepted
     */
    Index convert(const TokenType &token) const; // convert str to idx(never add(no ))

    /** convert index to str (ban unk-index).
     *  @param idx Index
     *  @exception: 1. domain_error, if use unk_idx as parameter when has_set_unk
     *              2. out_of_range, index not in range
     *  @return: 1. corresponding index, if : index is valid
     *           2. un-excepted(exception throw)
     */
    const TokenType& convert_ban_unk(Index idx) const; // if idx=unk, domain_error exception will be throw

    void set_unk();

    /** 
     *  get unk index(if unk not set, exception will be throw).
     *  @exception logic_error, if: !has_set_unk
     *  @eeturn  1. unk_idx, if: has_set_unk
     *           2. un-excepted(exception throw)
     */
    Index get_unk_idx() const;
    
    /**
     * get unk index(no exception throw).
     * @see set_unk()
     */
    Index get_unk_idx_without_throw() const noexcept;
    bool has_set_unk() const noexcept;
    bool is_unk_idx(Index idx) const noexcept;

    void freeze() noexcept;
    bool has_frozen() const noexcept;

    /**
     *   count the str occur times in dict ( 1/0)
     *   @return 1. 1, if str in dict
     *           2. 0, if not
     */
    std::size_t count(const TokenType &str) const noexcept;

    /**
     *  count with ban unk idx.
     *  @exception domain_error
     *  @return  1. 1
     *           2. 0
     *           3. un-excepted
     */
    std::size_t count_ban_unk(Index idx) const;

    void reset() noexcept;
    std::size_t size() const noexcept;
    std::size_t size_without_unk() const noexcept;

protected:
    /* dirived class may need query token2idx directly */
    const std::unordered_map<TokenType, Index>& get_token2idx_dict() const noexcept{ return token2idx; }
    std::string token2str(const TokenType &token) const noexcept{ return token2str_func(token); }

protected:
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);

private:
    std::unordered_map<TokenType, Index, Hash, KeyEqual> token2idx;
    std::vector<TokenType> idx2token;
    bool is_unk_seted;
    bool is_frozen;
    static constexpr Index UnkUnsetValue = -1;
    Index unk_idx;
    std::function<std::string(const TokenType&)> token2str_func;
};



template <typename TokenType, typename Hash=::std::hash<TokenType>,  typename KeyEqual=::std::equal_to<TokenType>>
class LookupTableWithCnt : public LookupTable<TokenType, Hash, KeyEqual>
{
    friend class boost::serialization::access;
public:
    LookupTableWithCnt(
        ::std::function<::std::string(const TokenType &)> token2str_func
        =static_cast<::std::string(*)(const TokenType&)>(&inner::token2str)) noexcept;
    // hidden the base class function with the same name
    Index convert(const TokenType &token);
    using LookupTable::convert; // look at C++ Primer (Chinese Version) P551. to make other `convert` is visitable

    /** 
     *  count str occurrence times (N).
     *  @return N  int, occurrence times
     */
    std::size_t count(const TokenType &token) const noexcept;
    std::size_t count_ban_unk(Index idx) const;

    void reset() noexcept;

protected:
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);

private:
    std::vector<unsigned> cnt;
};


template <typename TokenType, typename Hash=::std::hash<TokenType>, typename KeyEqual=::std::equal_to<TokenType>>
class LookupTableWithReplace : public LookupTableWithCnt<TokenType, Hash, KeyEqual>
{
    friend class boost::serialization::access;
public:
    LookupTableWithReplace(std::size_t seed = 1234, unsigned cnt_threshold = 1, float prob_threshold = 0.2f, 
        ::std::function<::std::string(const TokenType&)> token2str_func
        = static_cast<::std::string(*)(const TokenType&)>(&inner::token2str)) noexcept;
    template <typename Generator>
    LookupTableWithReplace(const Generator &rng, unsigned cnt_threshold = 1, float prob_threshold = 0.2f,
        ::std::function<::std::string(const TokenType&)> token2str_func
        = static_cast<::std::string(*)(const TokenType&)>(&inner::token2str)) noexcept;

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

template <typename TokenType, typename Hash,  typename KeyEqual>
LookupTable<TokenType, Hash, KeyEqual>::LookupTable(
    std::function<std::string(const TokenType&)> token2str_func) noexcept
    :is_unk_seted(false),
    is_frozen(false),
    unk_idx(UnkUnsetValue),
    token2str_func(token2str_func)
{}

template <typename TokenType, typename Hash,  typename KeyEqual>
typename LookupTable<TokenType, Hash, KeyEqual>::Index
LookupTable<TokenType, Hash, KeyEqual>::convert(const TokenType &token)
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
            else { throw std::out_of_range("key '" + token2str(token) + "' was not in LookupTable."); }
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

template <typename TokenType, typename Hash,  typename KeyEqual>
typename LookupTable<TokenType, Hash, KeyEqual>::Index
LookupTable<TokenType, Hash, KeyEqual>::convert(const TokenType &token) const
{
    // only read, no write(add)
    auto iter = token2idx.find(token);
    if( iter != token2idx.cend() ){ return iter->second; }
    else
    {
        if( has_set_unk() ){ return unk_idx; }
        else { throw std::out_of_range("key '" + token2str(token) + "' was not in LookupTable.") ; }
    }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
const TokenType&
LookupTable<TokenType, Hash, KeyEqual>::convert_ban_unk(Index idx) const
{
    if(idx == get_unk_idx_without_throw() && has_set_unk() )
    { 
        throw domain_error("unk index('" + to_string(idx) + "') was banned."); 
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

template <typename TokenType, typename Hash, typename KeyEqual>
void LookupTable<TokenType, Hash, KeyEqual>::set_unk()
{
    if( has_set_unk() ){ return; }
    if( !has_frozen() )
    {
        throw std::logic_error("before set unk, lookup table should be frozen firstly.");
    }
    else{ unk_idx = token2idx.size(); }
}

template <typename TokenType, typename Hash, typename KeyEqual>
typename LookupTable<TokenType, Hash, KeyEqual>::Index
LookupTable<TokenType, Hash, KeyEqual>::get_unk_idx() const
{
    if( !has_set_unk() ){ throw std::logic_error("unk was not set."); }
    else{ return unk_idx; }
}

template <typename TokenType, typename Hash, typename KeyEqual>
inline
typename LookupTable<TokenType, Hash, KeyEqual>::Index
LookupTable<TokenType, Hash, KeyEqual>::get_unk_idx_without_throw() const noexcept
{
    return unk_idx;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
bool LookupTable<TokenType, Hash, KeyEqual>::has_set_unk() const noexcept
{
    return unk_idx != UnkUnsetValue;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
bool LookupTable<TokenType, Hash, KeyEqual>::is_unk_idx(Index idx) const noexcept
{
    if( !has_set_unk() ){ return false; }
    else{ return idx == unk_idx; }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
void LookupTable<TokenType, Hash, KeyEqual>::freeze() noexcept
{
    is_frozen = true;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
bool LookupTable<TokenType, Hash, KeyEqual>::has_frozen() const noexcept
{
    return is_frozen;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
size_t LookupTable<TokenType, Hash, KeyEqual>::count(const TokenType& token) const noexcept
{
    return token2idx.count(token);
}

template <typename TokenType, typename Hash,  typename KeyEqual>
size_t LookupTable<TokenType, Hash, KeyEqual>::count_ban_unk(Index idx) const
{
    if( idx == get_unk_idx_without_throw() && has_set_unk() ){ throw std::domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else
    {
        return idx >= 0 && idx < size_without_unk();
    }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
void LookupTable<TokenType, Hash, KeyEqual>::reset() noexcept
{
    token2idx.clear();
    idx2token.clear();
    is_frozen = false;
    unk_idx = UnkUnsetValue;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
size_t LookupTable<TokenType, Hash, KeyEqual>::size() const noexcept
{
    if( has_set_unk() ){ return token2idx.size() + 1U; }
    else { return token2idx.size(); }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
inline
size_t LookupTable<TokenType, Hash, KeyEqual>::size_without_unk() const noexcept
{
    return token2idx.size();
}

template <typename TokenType, typename Hash,  typename KeyEqual>
template<class Archive>
void LookupTable<TokenType, Hash, KeyEqual>::serialize(Archive& ar, const unsigned int)
{
    ar &token2idx &idx2token &is_frozen &unk_idx;
}

/************************************
 *   Implementation  LookupTableWithCnt
 ************************************/

template <typename TokenType, typename Hash, typename KeyEqual>
LookupTableWithCnt<TokenType, Hash, KeyEqual>::LookupTableWithCnt(
    ::std::function<::std::string(const TokenType&)> token2str_func) noexcept 
    :LookupTable(token2str_func)
{}

template <typename TokenType, typename Hash,  typename KeyEqual>
typename LookupTableWithCnt<TokenType, Hash, KeyEqual>::Index
LookupTableWithCnt<TokenType, Hash, KeyEqual>::convert(const TokenType &token)
{
    Index idx = LookupTable::convert(token);
    if( !has_frozen() )
    {
        if( idx == static_cast<Index>(cnt.size()) )
        {
            // has added one!
            cnt.push_back(1);
        }
        else{ ++cnt[idx]; }
    }
    return idx;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
std::size_t LookupTableWithCnt<TokenType, Hash, KeyEqual>::count(const TokenType &token) const noexcept
{
    auto iter = get_token2idx_dict().find(token);
    if( iter == get_token2idx_dict().cend() ){ return 0U; }
    else{ return cnt[iter->second]; }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
std::size_t LookupTableWithCnt<TokenType, Hash, KeyEqual>::count_ban_unk(Index idx) const
{
    if( idx == get_unk_idx_without_throw() && has_set_unk() ){ throw std::domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else
    {
        if( idx >= 0 && idx < size_without_unk() ){ return cnt[idx]; }
        else{ return 0U; }
    }
}

template <typename TokenType, typename Hash,  typename KeyEqual>
void LookupTableWithCnt<TokenType, Hash, KeyEqual>::reset() noexcept
{
    LookupTable::reset();
    cnt.clear();
}

template <typename TokenType, typename Hash,  typename KeyEqual>
template<class Archive>
void LookupTableWithCnt<TokenType, Hash, KeyEqual>::serialize(Archive& ar, const unsigned int version)
{
    ar &cnt;
    LookupTable::serialize(ar, version);
}

/***************************************
 *   Implementation LookupTable with Replace
 ***************************************/
template <typename TokenType, typename Hash,  typename KeyEqual>
template <typename Generator>
LookupTableWithReplace<TokenType, Hash, KeyEqual>::LookupTableWithReplace(const Generator &rng, 
    unsigned cnt_threshold, float prob_threshold,
    ::std::function<::std::string(const TokenType &)> token2str_func) noexcept
    : LookupTableWithCnt(token2str_func),
    rand_generator(std::bind(std::uniform_real_distribution<float>(0, 1), rng)),
    cnt_threshold(cnt_threshold),
    prob_threshold(prob_threshold)
{}

template <typename TokenType, typename Hash,  typename KeyEqual>
LookupTableWithReplace<TokenType, Hash, KeyEqual>::LookupTableWithReplace(std::size_t seed, 
    unsigned cnt_threshold, float prob_threshold,
    ::std::function<::std::string(const TokenType &)> token2str_func) noexcept
    :LookupTableWithReplace(std::mt19937(seed), cnt_threshold, prob_threshold, token2str_func)
{}

template <typename TokenType, typename Hash,  typename KeyEqual>
void LookupTableWithReplace<TokenType, Hash, KeyEqual>::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept
{
    this->cnt_threshold = cnt_threshold;
    this->prob_threshold = prob_threshold;
}

template <typename TokenType, typename Hash,  typename KeyEqual>
typename LookupTableWithReplace<TokenType, Hash, KeyEqual>::Index
LookupTableWithReplace<TokenType, Hash, KeyEqual>::unk_replace_in_probability(Index idx) const
{
    if( !has_set_unk() ){ throw std::logic_error("unk was not set."); }
    else if( idx == get_unk_idx_without_throw() ){ return idx; }
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

template <typename TokenType, typename Hash,  typename KeyEqual>
template<class Archive>
void LookupTableWithReplace<TokenType, Hash, KeyEqual>::serialize(Archive& ar, const unsigned int version)
{
    ar &cnt_threshold &prob_threshold;
    LookupTableWithCnt::serialize(ar, version);
}

} // end of namespace lookup_table
using lookup_table::LookupTable;
using lookup_table::LookupTableWithCnt;
using lookup_table::LookupTableWithReplace;
} // end of namespace trivial
} // end of namespace slnn

#endif