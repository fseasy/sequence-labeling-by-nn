#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "cnn/dict.h"

#include "utils/stat.hpp"
#include "utils/typedeclaration.h"
#include "segmentor/cws_module/cws_tagging_system.h"

using namespace std ;
namespace po =  boost::program_options ;

void read_annotated_dataset(ifstream &is , cnn::Dict &tag_dict , vector<slnn::IndexSeq> &tag_seqs)
{
    unsigned line_cnt = 0;
    std::string line;
    std::vector<slnn::IndexSeq> tmp_tag_seqs;
    slnn::IndexSeq tag_seq;

    while (getline(is, line)) {
        if (0 == line.size()) continue;
        tag_seq.clear() ;
        std::istringstream iss(line) ;
        std::string words_line ;
        slnn::Seq tmp_word_cont,
            tmp_tag_cont ;
        while( iss >> words_line )
        {
            slnn::CWSTaggingSystem::parse_words2word_tag(words_line, tmp_word_cont, tmp_tag_cont) ;
            for( size_t i = 0 ; i < tmp_word_cont.size() ; ++i )
            {
                slnn::Index tag_id = tag_dict.Convert(tmp_tag_cont[i]) ;
                tag_seq.push_back(tag_id) ;
            }
        }
        tmp_tag_seqs.push_back(tag_seq);
        ++line_cnt;
        if(0 == line_cnt % 10000) { BOOST_LOG_TRIVIAL(info) << "reading " << line_cnt << " lines"; }
    }
    std::swap(tag_seqs, tmp_tag_seqs);
}

int main(int argc, char *argv[])
{
    po::options_description desc("Unit Test for CWS Evaluation");
    string gold_path, pred_path ;
    desc.add_options()
        ("gold_file", po::value<string>(), "path to gold file")
        ("pred_file", po::value<string>(), "path to predict file")
        ("help,h", "help") ;
    po::variables_map varmap ;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), varmap) ;
    po::notify(varmap) ;

    if( varmap.count("help") != 0 ||  varmap.count("gold_file") == 0 || varmap.count("pred_file") == 0 )
    {
        cerr << desc << "\n" ;
        return 1 ;
    }

    ifstream gold_f(gold_path), pred_f(pred_path) ;

    cnn::Dict tag_dict ;
    vector<slnn::IndexSeq> gold_seqs, pred_seqs ;
    read_annotated_dataset(gold_f, tag_dict, gold_seqs) ;
    read_annotated_dataset(pred_f, tag_dict, pred_seqs) ;
    
    slnn::CWSTaggingSystem tag_sys ;
    tag_sys.build(tag_dict) ;

    slnn::CWSStat stat(tag_sys) ;

    array<cnn::real , 4> eval_rst = stat.eval(gold_seqs, pred_seqs) ;

    cnn::real acc = eval_rst[0],
        p = eval_rst[1],
        r = eval_rst[2],
        f1 = eval_rst[3] ;

    cerr << "ACC = " << acc
        << "\nP = " << p
        << "\nR = " << r
        << "\nF1 = " << f1
        << "\n" ;
    return 0 ;
}
