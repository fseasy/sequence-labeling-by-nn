#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std ;
using namespace cnn ;

#define EXIT_WITH_INFO(info) do{ std::cerr << __FILE__ << "," <<  __LINE__ \\
                                      << "," << __func__  << " : "\\
                                      << info << std::endl ; abort() ; }while(0) ;

using InstancePair = pair<vector<int> , vector<int>>  ;

const string POS_TRAIN_PATH = "/data/ltp/ltp-data/pos/pku-weibo-train.pos"
const string POS_DEV_PATH = "/data/ltp/ltp-data/pos/pku-weibo-holdout.pos"
const string POS_TEST_PATH = "/data/ltp/ltp-data/pos/pku-weibo-test.pos"

// almost copy from : http://stackoverflow.com/a/9676623/4869018
vector<string>& split(const string &str , vector<string> &container , const string &delimiter="\t " , bool skip_empty=true)
{
    vector<string> tmp_cont ;
    string::size_type start_pos = 0 ,
                      delim_pos ;
    while( ( delim_pos = str.find_first_of(delimiter , start_pos) ) != string::npos )
    {
        if( start_pos != delim_pos || ! skip_empty ) // not empty or not skip , we'll add the slice ; 
                                                     // => only Empty && skip , we'll ignore the slice !
            tmp_cont.emplace_back(str , start_pos , delim_pos - start_pos) ;
        start_pos = delim_pos + 1 ;
    }
    if(start_pos < str.length()) // delim pos must be string::npos , but start_pos may be has out of range( == str.length() )
        tmp_cont.emplace_back(str , start_pos) ;
    tmp_cont.swap(container) ;
    return container ;
}

string trim(const string & str , const string &spaces=" \t\f\n\r\v" )
{
    string::size_type start_pos = str.find_fisrt_not_of(spaces) ,
                      end_pos = str.find_last_not_of(spaces) ;
    // 1. normal situation , that is start_pos and end_pos is all indices of str , no problem 
    // 2. str is all space , start_pos and end_pos is all string::npos , should do specifically
    return (start_pos != string::npos) ? str.substr(start_pos , end_pos - start_pos + 1) : string("") ;
}

template<typename Iterator>
void print(Iterator begin , Iterator end)
{
    for(Iterator i = begin ; i != end ; ++i)
    {
        cout << *i << "\t" ;
    }
    cout << endl ;
}



void read_dataset_and_build_dicts(istream &is , vector<InstancePair> & samples  , cnn::Dict & sent_dict , cnn::Dict & tag_dict)
{
    string line ;
    vector<InstancePair> tmp_samples ;
    while(getline(is , line))
    {
        line = trim(line) ;
        vector<string> strpair_cont ;
        split(line , strpair_cont , "\t") ;
        vector<int> sent , tag_seq ;
        vector<string> word_and_tag ;
        for(string &strpair : strpair_cont)
        {
            word_and_tag.clear() ;
            split(strpair , word_and_tag , "_") ;
            assert( 2 == word_and_tag.size()) ;
            int word_id = sent_dict.Convert(word_and_tag[0]) ;
            int tag_id = tag_dict.Convert(word_and_tag[1]) ;
            sent.push_back(word_id) ;
            tag_seq.push_back(tag_id) ;
        }
        tmp_samples.emplace_back(sent , tag_seq) ; // using `pair` construction pair(first_type , second_type)
    }
    tmp_samples.swap(samples) ;
}


int main(int argc , char *argv[])
{
    ifstream train_is(POS_TRAIN_PATH) ;
    if(!train_is) EXIT_WITH_INFO( (string("failed to open file" + string(POS_TRAIN_PATH))).c_str()) ;
    cnn::Dict sent_dict , tag_dict ;
    vector<InstancePair> samples ;
    read_dataset_and_build_dicts(train_is , samples , sent_dict , tag_dict) ;
    train_is.close() ;
    return 0 ;
}
