#include "lookup_table.h"
using namespace std;

namespace slnn{
namespace trivial{
namespace lookup_table{

// do template instantiation.
template class LookupTable<char32_t>;
template class LookupTable<u32string>;
template class LookupTable<string>;

template class LookupTableWithCnt<char32_t>;
template class LookupTableWithCnt<u32string>;
template class LookupTableWithCnt<string>;

template class LookupTableWithReplace<char32_t>;
template class LookupTableWithReplace<u32string>;
template class LookupTableWithReplace<string>;


} // end of namespace lookup_table
} // end of namespace trivial
} // end of namespace slnn
