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

template<typename Iterator>
void print(Iterator begin , Iterator end)
{
    for(Iterator i = begin ; i != end ; ++i)
    {
        cout << *i << "\t" ;
    }
    cout << endl ;
}

int main(int argc , char *argv[])
{
    string test_str = "I have a dream!" ;
    vector<string> con ; 
    split(test_str , con) ;
    print(con.begin() , con.end() ) ;

    return 0 ;
}
