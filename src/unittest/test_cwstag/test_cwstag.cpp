#define CATCH_CONFIG_MAIN
#include <iostream>
#include <cmath>
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "segmenter/cws_module/cws_eval.h"
#include "utils/typedeclaration.h"
#include "../3rdparty/catch/include/catch.hpp"

using namespace std;
using namespace slnn;
using namespace slnn::segmenter;

TEST_CASE("tagseq2wordrange", "[range]")
{
    vector<Index> valid_tagseq{ 
        Tag::TAG_B_ID, Tag::TAG_E_ID,
        Tag::TAG_S_ID,
        Tag::TAG_B_ID, Tag::TAG_M_ID, Tag::TAG_E_ID };
    vector<pair<unsigned, unsigned>> valid_range{
        make_pair(0,1),
        make_pair(2,2),
        make_pair(3,5)
    };
    bool is_equal = token_module::tagseq2word_range_list(valid_tagseq) == valid_range;
    REQUIRE(is_equal);
    is_equal = token_module::not_valid_tagseq2word_range_list(valid_tagseq) == valid_range;
    REQUIRE(is_equal);

    vector<Index> not_valid_tagseq1{
        Tag::TAG_M_ID, Tag::TAG_E_ID, // start from M
        Tag::TAG_B_ID, Tag::TAG_S_ID, // B->S
        Tag::TAG_B_ID, Tag::TAG_B_ID, // B->B
        Tag::TAG_M_ID, Tag::TAG_B_ID, // M->B
        Tag::TAG_M_ID, Tag::TAG_S_ID, // M->S
        Tag::TAG_E_ID, Tag::TAG_E_ID, // E->E
        Tag::TAG_E_ID, Tag::TAG_M_ID, // E->M
        Tag::TAG_S_ID, Tag::TAG_M_ID, // S->M
        Tag::TAG_S_ID, Tag::TAG_E_ID, // S->E
        Tag::TAG_M_ID                 // end with M
    };
    vector<pair<unsigned, unsigned>> not_valid_tagseq1_range{
        make_pair(0, 1),
        make_pair(2, 2), make_pair(3, 3),
        make_pair(4, 4), make_pair(5, 6),
        make_pair(7, 8), make_pair(9, 9),
        make_pair(10, 10), make_pair(11, 11),
        make_pair(12, 12), make_pair(13, 13),
        make_pair(14, 14), make_pair(15, 15),
        make_pair(16, 16), make_pair(17, 17),
        make_pair(18, 18)
    };
    is_equal = token_module::not_valid_tagseq2word_range_list(not_valid_tagseq1) == not_valid_tagseq1_range;
    for( auto& p : token_module::not_valid_tagseq2word_range_list(not_valid_tagseq1) )
    {
        cerr << "(" << p.first << ", " << p.second << ") ";
    }
    cerr << "\n";
    REQUIRE(is_equal);
    vector<Index> not_valid_tagseq2{
        Tag::TAG_E_ID, // start from E
        Tag::TAG_B_ID // end with B
    };
    vector<pair<unsigned, unsigned>> not_valid_tagseq2_range{
        make_pair(0, 0),
        make_pair(1, 1)
    };
    is_equal = token_module::not_valid_tagseq2word_range_list(not_valid_tagseq2) == not_valid_tagseq2_range;
    REQUIRE(is_equal);
}

TEST_CASE("tagseq2wordlist", "[word]")
{
    vector<Index> valid_tagseq{
        Tag::TAG_S_ID,
        Tag::TAG_B_ID, Tag::TAG_E_ID,
        Tag::TAG_B_ID, Tag::TAG_M_ID, Tag::TAG_E_ID
    };
    u32string charseq = U"ABCDEF";
    vector<u32string> word_list{ U"A", U"BC", U"DEF" };
    REQUIRE(token_module::generate_wordseq_from_chartagseq(charseq, valid_tagseq) == word_list);

    vector<Index> not_valid_tagseq = {
        Tag::TAG_E_ID,
        Tag::TAG_M_ID, Tag::TAG_B_ID,
        Tag::TAG_M_ID, Tag::TAG_S_ID,
        Tag::TAG_E_ID, Tag::TAG_M_ID,
        Tag::TAG_E_ID, Tag::TAG_E_ID,
        Tag::TAG_S_ID, Tag::TAG_M_ID,
        Tag::TAG_S_ID, Tag::TAG_E_ID,
        Tag::TAG_B_ID, Tag::TAG_B_ID,
        Tag::TAG_B_ID, Tag::TAG_S_ID,
        Tag::TAG_M_ID
    };
    u32string charseq4not_valid = U"ABCDEFGHIJKLMNOPQR";
    vector<u32string> wordseq4not_valid = {
        U"A",
        U"B", U"CD",
        U"E",
        U"F", U"GH",
        U"I",
        U"J", U"K",
        U"L", U"M",
        U"N", U"O",
        U"P", U"Q",
        U"R"
    };
    REQUIRE(token_module::generate_wordseq_from_not_valid_chartagseq(charseq4not_valid, not_valid_tagseq) 
        == wordseq4not_valid);
}


TEST_CASE("eval", "[eval]")
{
    vector<Index> gold_tagseq = {
        Tag::TAG_S_ID, Tag::TAG_B_ID, Tag::TAG_E_ID, Tag::TAG_B_ID, Tag::TAG_E_ID, Tag::TAG_S_ID
    };
    vector<Index> pred_tagseq = {
        Tag::TAG_S_ID, Tag::TAG_B_ID, Tag::TAG_M_ID, Tag::TAG_E_ID, Tag::TAG_S_ID, Tag::TAG_S_ID
    };
    eval::SegmenterEval eval_ins;
    eval_ins.start_eval();
    eval_ins.eval_iteratively(gold_tagseq, pred_tagseq);
    auto eval_result = eval_ins.end_eval();
    REQUIRE(abs(eval_result.f1 - 50.f) < 0.0001f);
}