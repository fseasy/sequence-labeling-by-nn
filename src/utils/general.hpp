#ifndef GENERAL_H_INCLUDED_
#define GENERAL_H_INCLUDED_

#include <fstream>
#include <sys/stat.h>
#include <string>

#ifndef __linux     // WINDOWS
    #include <io.h>
    #define access _access_s
    #define F_OK 0
    #define R_OK 4
    #define W_OK 2
    #define X_OK 1
#else               // LINUX
    #include <unistd.h>
#endif

#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>

namespace slnn
{

struct FileUtils
{
    static 
    bool exists(const std::string &file_name)
    {
        struct stat file_status ;
        return stat(file_name.c_str() , &file_status) == 0 ;
    }
    
    static
    bool writeable(const std::string &file_name)
    {
        if(exists(file_name)) return access(file_name.c_str() , W_OK) == 0 ; 
        else
        {
            // we thought it is a dir_name + file_name
            std::string::size_type split_pos = file_name.find_last_of("/\\") ;
            std::string dir_name = file_name.substr(0 , split_pos) ;
            return access(dir_name.c_str() , W_OK) == 0 ;
        }
    }

} ;

void fatal_error(const std::string &exit_msg)
{
        BOOST_LOG_TRIVIAL(fatal) << exit_msg << "\n"
            "Exit!" ;
        exit(1) ;
}

void varmap_key_fatal_check(boost::program_options::variables_map &var_map , const std::string &key , const std::string &exit_msg)
{
    if(0 == var_map.count(key)) fatal_error(exit_msg) ;
}

} // end of namespace slnn

#endif
