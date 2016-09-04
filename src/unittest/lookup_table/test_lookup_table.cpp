#define CATCH_CONFIG_MAIN
#include "trivial/lookup_table.h"
#include "thirdparty/catch/include/catch.hpp"

using namespace std;
using namespace slnn;

TEST_CASE("LookupTable", "[LookupTable]")
{
    LookupTable lookup_table;
    
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
    REQUIRE(lookup_table.convert(word1_idx) == word1);
    REQUIRE_THROWS_AS(lookup_table.convert(1), out_of_range);

    // add another word
    auto word2_idx = lookup_table.convert(word2);

    REQUIRE(lookup_table.size() == 2U);
    REQUIRE(lookup_table.has_frozen() == false);
    REQUIRE(lookup_table.has_set_unk() == false);
    REQUIRE(lookup_table.count(word2) == 1U);
    REQUIRE(lookup_table.count(word2_idx) == 1U);
    REQUIRE(word2_idx == 0);
    REQUIRE(lookup_table.convert(word2_idx) == word2);
    REQUIRE_THROWS_AS(lookup_table.convert(1), out_of_range);

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
    lookup_table.set_unk();
    REQUIRE(lookup_table.has_set_unk() == true);
    REQUIRE_NOTHROW(lookup_table.get_unk_idx());
    REQUIRE(lookup_table.is_unk_idx(lookup_table.get_unk_idx()) == true);
    REQUIRE(lookup_table.count(lookup_table.get_unk_idx()) == 1U);
    REQUIRE(lookup_table.count_ban_unk(lookup_table.get_unk_idx()));
    REQUIRE(lookup_table.size() == 3U);
    REQUIRE(lookup_table.size_without_unk() == 2U);
    REQUIRE(lookup_table.convert("never_occur") == lookup_table.get_unk_idx());
    REQUIRE_NOTHROW(lookup_table.convert(lookup_table.get_unk_idx()));
    REQUIRE_THROWS_AS(lookup_table.convert_ban_unk(lookup_table.get_unk_idx()), domain_error);
}