#ifndef GENERAL_H_INCLUDED_
#define GENERAL_H_INCLUDED_
#include <memory>
#include <algorithm>
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
            std::string dir_name ;
            std::string::size_type split_pos = file_name.find_last_of("/\\") ;
            if(split_pos == std::string::npos){ dir_name = "."; } // current directory
            else { dir_name = file_name.substr(0 , split_pos); }
            return access(dir_name.c_str() , W_OK) == 0 ;
        }
    }

} ;

void fatal_error(const std::string &exit_msg)
{
        BOOST_LOG_TRIVIAL(fatal) << exit_msg << "\n"
            "Exit!" ;
#if (defined(_WIN32)) &&  (_DEBUG)
        system("pause");
#endif
        exit(1) ;
}

void varmap_key_fatal_check(boost::program_options::variables_map &var_map , const std::string &key , const std::string &exit_msg)
{
    if(0 == var_map.count(key)) fatal_error(exit_msg) ;
}

void build_cnn_parameters(const std::string &program_name, unsigned cnn_mem, int &cnn_argc, std::shared_ptr<char *> &cnn_argv_sp)
{
    char * program_name_cstr = new char[program_name.length() + 1](); // value initialization
    std::copy(std::begin(program_name), std::end(program_name), program_name_cstr);
    
    auto deleter = [&cnn_argc](char **ptr)
    {
        for( int i = 0 ; i < cnn_argc ; ++i ) { delete[] ptr[i]; }
    } ;
    if( cnn_mem != 0 )
    {
        std::string cnn_mem_key = "--cnn-mem";
        char * cnn_mem_key_cstr = new char[cnn_mem_key.length() + 1]();
        std::copy(std::begin(cnn_mem_key), std::end(cnn_mem_key), cnn_mem_key_cstr);

        std::string cnn_mem_value = "";
        cnn_mem_value = std::to_string(cnn_mem);
        char * cnn_mem_value_cstr = new char[cnn_mem_value.length() + 1]();
        std::copy(std::begin(cnn_mem_value), std::end(cnn_mem_value), cnn_mem_value_cstr);
        
        cnn_argc = 3 ; // program_name --cnn-mem [mem_vlaue] NULL (Attention : argv should has anothre nullptr !)
        char **cnn_argv_cstr = new char*[cnn_argc+1]{ program_name_cstr, cnn_mem_key_cstr, cnn_mem_value_cstr, nullptr };
        cnn_argv_sp = std::shared_ptr<char *>(cnn_argv_cstr, deleter);
    }
    else
    {
        cnn_argc = 1 ;
        char **cnn_argv_cstr = new char*[cnn_argc + 1]{ program_name_cstr, nullptr };
        cnn_argv_sp = std::shared_ptr<char *>(cnn_argv_cstr, deleter);
    }
}

} // end of namespace slnn

#endif
