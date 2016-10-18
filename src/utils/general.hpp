#ifndef GENERAL_H_INCLUDED_
#define GENERAL_H_INCLUDED_
#include <memory>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <cstdio>

#ifndef __linux     // WINDOWS
    #include <io.h>
    #define io_access _access_s  // name `access` is confilicate with boost::serialization::access 
    #define F_OK 0
    #define R_OK 4
    #define W_OK 2
    #define X_OK 1
#else               // LINUX
    #include <unistd.h>
    #define io_access access
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
        if(exists(file_name)) return io_access(file_name.c_str() , W_OK) == 0 ; 
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

void build_dynet_parameters(const std::string &program_name, unsigned dynet_mem, int &dynet_argc, std::shared_ptr<char *> &dynet_argv_sp)
{
    char * program_name_cstr = new char[program_name.length() + 1](); // value initialization
    //std::copy(std::begin(program_name), std::end(program_name), program_name_cstr);
    strncpy(program_name_cstr, program_name.c_str(), program_name.length() + 1);
    auto deleter = [&dynet_argc](char **ptr)
    {
        for( int i = 0 ; i < dynet_argc ; ++i ) { delete[] ptr[i]; }
    } ;
    if( dynet_mem != 0 )
    {
        std::string dynet_mem_key = "--dynet-mem";
        char * dynet_mem_key_cstr = new char[dynet_mem_key.length() + 1]();
        //std::copy(std::begin(dynet_mem_key), std::end(dynet_mem_key), dynet_mem_key_cstr);
        strncpy(dynet_mem_key_cstr, dynet_mem_key.c_str(), dynet_mem_key.length() + 1);

        std::string dynet_mem_value = "";
        dynet_mem_value = std::to_string(dynet_mem);
        char * dynet_mem_value_cstr = new char[dynet_mem_value.length() + 1]();
        //std::copy(std::begin(dynet_mem_value), std::end(dynet_mem_value), dynet_mem_value_cstr);
        strncpy(dynet_mem_value_cstr, dynet_mem_value.c_str(), dynet_mem_value.length() + 1);

        const int const_dynet_argc = 3 ; // program_name --dynet-mem [mem_vlaue] NULL (Attention : argv should has anothre nullptr !)
        char **dynet_argv_cstr = new char*[const_dynet_argc+1]{ program_name_cstr, dynet_mem_key_cstr, dynet_mem_value_cstr, nullptr };
        dynet_argc = const_dynet_argc;
        dynet_argv_sp = std::shared_ptr<char *>(dynet_argv_cstr, deleter);
    }
    else
    {
        const int const_dynet_argc = 1 ;
        char **dynet_argv_cstr = new char*[const_dynet_argc + 1]{ program_name_cstr, nullptr };
        dynet_argc = const_dynet_argc;
        dynet_argv_sp = std::shared_ptr<char *>(dynet_argv_cstr, deleter);
    }
}

} // end of namespace slnn

#endif
