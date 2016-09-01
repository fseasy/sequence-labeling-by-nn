#include "lookup_table.h"
using namespace std;

namespace slnn{

LookupTable::LookupTable()
    :is_unk_seted(false),
    is_frozen(false),
    unk_idx(-1)
{}

LookupTable::Index 
LookupTable::convert(const std::string &str)
{
    auto iter = str2idx.find(str);
    if( iter != str2idx.cend() )
    { 
        return iter->second; 
    } 
    else
    {
        // not find
        if( has_frozen() )
        { 
            if( has_set_unk() ){ return unk_idx; }
            else { throw out_of_range("key '" + str + "' was not in LookupTable."); }
        }
        else
        {
            // add new key
            Index nextIndex = str2idx.size();
            str2idx[str] = nextIndex; // add str -> idx
            idx2str.push_back(str); // add idx -> str
            return nextIndex;
        }
    }
}

LookupTable::Index 
LookupTable::convert(const std::string &str) const
{
    // only read, no write(add)
    auto iter = str2idx.find(str);
    if( iter == str2idx.cend() )
    { 
        if( has_set_unk() ){ return unk_idx; }
        else { throw out_of_range("key '" + str + "' was not in LookupTable.") }
    }
    else{ return iter->second; }
}

const std::string& 
LookupTable::convert(Index idx) const
{
    if( idx >= size() ){ throw out_of_range("index '" + to_string(idx) + "' was out of range( size = " + 
                                                     to_string(size())+")")}
    else{ return idx2str[idx]; }
}

const std::string&
LookupTable::convert_ban_unk(Index idx) const
{
    if( has_set_unk() && idx == unk_idx ){ throw domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else return convert(idx);
}

void LookupTable::set_unk()
{
    if( has_set_unk() ){ return; }
    if( has_frozen() )
    {
        unfreeze();
        unk_idx = convert(UnkStr);
        freeze();
    }
    else{ unk_idx = convert(UnkStr); }
    is_unkseted = true;
}
Index LookupTable::get_unk_idx()
{
    if( !has_set_unk() ){ throw logic_error("unk was not set."); }
    else{ return unk_idx; }
}
bool LookupTable::has_set_unk() const
{
    return is_unkseted;
}
bool LookupTable::is_unk_idx(Index idx) const
{
    if( !has_set_unk() ){ return false; }
    else{ return idx == unk_idx; }
}

void LookupTable::freeze()
{
    is_frozen = true;
}
void LookupTable::unfreeze()
{
    is_frozen = false;
}
bool LookupTable::has_frozen() const
{
    return is_frozen;
}

size_t LookupTable::count(const string& str) const
{
    auto iter = str2idx.find(str);
    if( iter == str2idx.cend() ){ return 0U; }
    else{ return 1U; }
}
size_t LookupTable::count(Index idx) const
{
    if( idx >= 0 && idx < static_cast<int>(size()) ){ return 1U; }
    else{ return 0U; }
}
size_t LookupTable::count_ban_unk(Index idx) const
{
    if( has_set_unk() && idx == unk_idx ){ throw domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else { return count(idx); }
}

void LookupTable::reset()
{
    str2idx.clear();
    idx2str.clear();
    is_unk_seted = false;
    is_frozen = false;
    unk_idx = -1;
}

size_t LookupTable::size() const
{
    return str2idx.size();
}
size_t LookupTable::size_without_unk() const
{
    if( has_set_unk() ){ return size() - 1; }
    else { return size();  }
}

/*------------ LookupTable with Count ------------------*/

LookupTableWithCnt::Index
LookupTableWithCnt::convert(const string& str)
{
    Index idx = LookupTable::convert(str);
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

std::size_t LookupTableWithCnt::count(const std::string &str) const
{
    auto iter = str2idx.find(str);
    if( iter == str2idx.cend() ){ return 0U; }
    else{ return cnt[iter->second]; }
}
std::size_t LookupTableWithCnt::count(Index idx) const
{
    if( idx >= 0 && idx < size() ){ return cnt[idx]; }
    else{ return 0U; }
}
std::size_t LookupTableWithCnt::count_ban_unk(Index idx) const
{
    if( has_set_unk() && idx == unk_idx ){ throw domain_error("unk index('" + to_string(idx) + "') was banned."); }
    else{ return count(idx); }
}

void LookupTableWithCnt::reset()
{
    LookupTable::reset();
    cnt.clear();
}

/*---------------- LookupTableWithReplace -------------------*/

LookupTableWithReplace::LookupTableWithReplace(std::size_t seed, unsigned cnt_threshold, float prob_threshold)
    :LookupTableWithReplace(mt19937(seed), cnt_threshold, prob_threshold)
{}

void LookupTableWithReplace::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold)
{
    this->cnt_threshold = cnt_threshold;
    this->prob_threshold = prob_threshold;
}

LookupTableWithReplace::Index
LookupTableWithReplace::unk_replace_in_probability(Index idx)
{
    if( !has_set_unk() ){ throw logic_error("unk was not set."); }
    else if( idx == get_unk_idx() ){ return idx; }
    else if( idx < 0 || idx >= size() ){ throw out_of_range("index '" + to_string(idx) + 
                                         "' was out of range( size = " + to_string(size())+")")}
    else
    {
        if( count(idx) <= cnt_threshold && rand_generator() <= prob_threshold ){ return get_unk_idx(); }
        else{ return idx; }
    }
}

} // end of namespace slnn