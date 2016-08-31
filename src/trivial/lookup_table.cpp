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
    if( iter != str2idx.cend() ){ return iter->second; }
    else
    {
        // not find
        if( has_frozen() )
        {
            throw out_of_range("key '" + str + "' was not in LookupTable");
        }
    }
}

LookupTable::Index 
LookupTable::convert(const std::string &str) const
{

}

} // end of namespace slnn