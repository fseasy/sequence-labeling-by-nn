#ifndef SLNN_TRIVIAL_LOOKUP_TABLE_H_
#define SLNN_TRIVIAL_LOOKUP_TABLE_H_
#include <unordered_map>
#include <string>
namespace slnn{

class LookupTable
{
public:
    using Index = int; // we should use the global declaration. But it seems so duplicted.
public:
    LookupTable();
    
    Index convert(const std::string &str);
    Index convert(const std::string &str) const;
    const std::string& Convert(Index idx) const;
    
    void set_unk();
    void unset_unk();
    bool has_set_unk() const;
    
    void freeze();
    void unfreeze();
    bool has_frozen() const;

    bool count(const std::string &str) const;
    
    void reset();
    std::size_t size() const;


private:
    std::unordered_map<std::string, Index> str2idx;
    std::vector<std::string> idx2str;
    std::vector<unsigned> cnt;
    bool is_unk_seted;
    bool is_frozen;
    static constexpr std::string UnkStr;
    Index unk_idx;
};

LookupTable::UnkStr = "*u*n*k*";

} // end of namespace slnn

#endif