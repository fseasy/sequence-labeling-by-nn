// #define CATCH_CONFIG_MAIN
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "trivial/lookup_table/lookup_table.h"
//#include "../3rdparty/catch/include/catch.hpp"

using namespace std;
using namespace slnn;

template class LookupTable<char32_t>;

int main()
{

    LookupTable<string> lookup_table;
    LookupTableWithCnt<char32_t> l2;
    LookupTableWithReplace<u32string> l3;
}

/*
TEST_CASE("LookupTable", "[LookupTable]")
{
    LookupTable<string> lookup_table;
    
    // check init status
    REQUIRE(lookup_table.size() == 0U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);

    const string word2 = "word2";
    const string word1 = "word1";
    // add 1 word
    auto word1_idx = lookup_table.convert(word1);
    
    REQUIRE(lookup_table.size() == 1U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);
    REQUIRE(lookup_table.count(word1) == 1U);
    REQUIRE(lookup_table.count(word1_idx) == 1U);
    REQUIRE(lookup_table.count(word2) == 0U);
    REQUIRE(lookup_table.count(1) == 0U);
    REQUIRE(word1_idx == 0);
    REQUIRE(lookup_table.convert_ban_unk(word1_idx) == word1);
    REQUIRE_THROWS_AS(lookup_table.convert_ban_unk(1), out_of_range);

    // add another word
    auto word2_idx = lookup_table.convert(word2);

    REQUIRE(lookup_table.size() == 2U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);
    REQUIRE(lookup_table.count(word2) == 1U);
    REQUIRE(lookup_table.count(word2_idx) == 1U);
    REQUIRE(word2_idx == 1);
    REQUIRE(lookup_table.convert_ban_unk(word2_idx) == word2);
    REQUIRE_THROWS_AS(lookup_table.convert_ban_unk(2), out_of_range);

    // add duplicated word
    auto dup_idx = lookup_table.convert(word2);
    REQUIRE(lookup_table.size() == 2U);
    REQUIRE(lookup_table.count(dup_idx) == 1U);

    // freeze
    lookup_table.freeze();
    REQUIRE(lookup_table.has_frozen() == true);
    REQUIRE(lookup_table.convert(word2) == word2_idx);
    REQUIRE_THROWS_AS(lookup_table.convert("never_occur"), out_of_range);
    
    // unk
    REQUIRE(lookup_table.has_set_unk() == false);
    REQUIRE_THROWS_AS(lookup_table.get_unk_idx(), logic_error);
    REQUIRE_THROWS_AS(lookup_table.convert("never_occur"), out_of_range);
    lookup_table.set_unk();
    REQUIRE(lookup_table.has_set_unk() == true);
    REQUIRE_NOTHROW(lookup_table.get_unk_idx());
    REQUIRE(lookup_table.is_unk_idx(lookup_table.get_unk_idx()) == true);
    REQUIRE(lookup_table.count(lookup_table.get_unk_idx()) == 1U);
    REQUIRE_THROWS_AS(lookup_table.count_ban_unk(lookup_table.get_unk_idx()), domain_error);
    REQUIRE(lookup_table.size() == 3U);
    REQUIRE(lookup_table.size_without_unk() == 2U);
    REQUIRE(lookup_table.convert("never_occur") == lookup_table.get_unk_idx());
    REQUIRE_NOTHROW(lookup_table.convert_ban_unk(word2_idx));
    REQUIRE_THROWS_AS(lookup_table.convert_ban_unk(lookup_table.get_unk_idx()), domain_error);

    // serialize
    stringstream ss;
    boost::archive::text_oarchive to(ss);
    to << lookup_table;
    boost::archive::text_iarchive ti(ss);
    LookupTable<string> lookup_table_copy;
    ti >> lookup_table_copy;
    REQUIRE(lookup_table_copy.size() == lookup_table.size());
    REQUIRE(lookup_table_copy.convert(word1) == word1_idx);
    REQUIRE(lookup_table_copy.convert_ban_unk(word2_idx) == word2);
    REQUIRE(lookup_table_copy.has_frozen() == lookup_table.has_frozen());
    REQUIRE(lookup_table_copy.has_set_unk() == lookup_table.has_set_unk());

    // reset
    lookup_table.reset();
    REQUIRE(lookup_table.size() == 0U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);
}


TEST_CASE("LookupTableWithCnt", "[LookupTable]")
{
    LookupTableWithCnt<string> lookup_table;

    // check init status
    REQUIRE(lookup_table.size() == 0U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);

    string word1 = "word1";
    string word2 = "word2";
    // check count
    auto word1_idx1 = lookup_table.convert(word1);
    auto word1_idx2 = lookup_table.convert(word1);
    REQUIRE(word1_idx1 == word1_idx2);
    lookup_table.convert(word1);
    REQUIRE(lookup_table.count_ban_unk(word1_idx1) == 3U);
    REQUIRE(lookup_table.count(word1) == 3U);
    lookup_table.freeze();
    lookup_table.convert(word1);
    REQUIRE(lookup_table.count(word1) == 3U);
    REQUIRE(lookup_table.count("never_occur") == 0U);
    lookup_table.set_unk();
    REQUIRE(lookup_table.get_unk_idx() == 1);
    REQUIRE(lookup_table.count_ban_unk(word1_idx1) == 3U);
    REQUIRE_THROWS_AS(lookup_table.count_ban_unk(lookup_table.get_unk_idx()), domain_error);

    // check serialization
    stringstream ss;
    boost::archive::text_oarchive to(ss);
    to << lookup_table;
    boost::archive::text_iarchive ti(ss);
    LookupTableWithCnt<string> lookup_table_copy;
    ti >> lookup_table_copy;
    REQUIRE(lookup_table_copy.size() == lookup_table.size());
    REQUIRE(lookup_table_copy.count(word1) == 3U);
    REQUIRE(lookup_table_copy.has_frozen() == lookup_table.has_frozen());
    REQUIRE(lookup_table_copy.has_set_unk() == lookup_table.has_set_unk());
}


TEST_CASE("LookupTableWithReplace", "[LookupTable]")
{
    LookupTableWithReplace<string> lookup_table;

    // check init status
    REQUIRE(lookup_table.size() == 0U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);

    string word1 = "word1";
    auto word1_idx = lookup_table.convert(word1);
    
    // check replace
    REQUIRE_THROWS_AS(lookup_table.unk_replace_in_probability(word1_idx), logic_error);
    lookup_table.set_unk();
    int replace_cnt = 0;
    for( int i = 1000; i > 0; --i )
    {
        if( lookup_table.unk_replace_in_probability(word1_idx) == lookup_table.get_unk_idx() ){ ++replace_cnt; }
    }
    REQUIRE(replace_cnt > 0);
    replace_cnt = 0;
    for( int i = 1000; i > 0; --i )
    {
        if( lookup_table.unk_replace_in_probability(lookup_table.get_unk_idx()) == lookup_table.get_unk_idx() )
        { 
            ++replace_cnt; 
        }
    }
    REQUIRE(replace_cnt == 1000);
    REQUIRE_THROWS_AS(lookup_table.unk_replace_in_probability(2), out_of_range);
    REQUIRE_THROWS_AS(lookup_table.unk_replace_in_probability(-2), out_of_range);
    lookup_table.set_unk_replace_threshold(lookup_table.get_cnt_threshold(), 1.f);
    replace_cnt = 0;
    for( int i = 1000; i > 0; --i )
    {
        if( lookup_table.unk_replace_in_probability(word1_idx) == lookup_table.get_unk_idx() )
        { 
            ++replace_cnt; 
        }
    }
    REQUIRE(replace_cnt == 1000);
    lookup_table.convert(word1);
    replace_cnt = 0;
    for( int i = 1000; i > 0; --i )
    {
        if( lookup_table.unk_replace_in_probability(word1_idx) == lookup_table.get_unk_idx() )
        { 
            ++replace_cnt; 
        }
    }
    REQUIRE(replace_cnt == 0);
}
*/