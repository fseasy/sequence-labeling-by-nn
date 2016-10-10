#ifndef SLNN_TRIVIAL_CHARCODE_DETECTOR_H_
#define SLNN_TRIVIAL_CHARCODE_DETECTOR_H_
#include <fstream>
#include <iostream>
#include "charcode_base.hpp"

namespace slnn{
namespace charcode{
namespace detector{
/**
 * encoding type detector.
 * TODO.
 * singleton design pattern for C++, see : http://stackoverflow.com/questions/1008019/c-singleton-design-pattern
 */
class EncodingDetector{
public:
    EncodingDetector(const EncodingDetector&) = delete; // ban copy
    void operator=(const EncodingDetector&) = delete; // ban copy assign
public:
    static EncodingDetector* get_detector() noexcept;
    static base::EncodingType get_console_encoding() noexcept;

public:
    void set_encoding(base::EncodingType) noexcept;
    void set_encoding(std::string &encoding_name) noexcept;
    base::EncodingType get_encoding() const noexcept;
    base::EncodingType detect_and_set_encoding(const std::string &bytes) noexcept;
    base::EncodingType detect_and_set_encoding(std::ifstream &f) noexcept; // reject iostream
    base::EncodingType detect_and_set_encoding(std::istream &stdinf) noexcept;
    base::EncodingType detect_and_set_encoding_from_fstream(std::ifstream &f) noexcept; // explicit to detect from file.
private:
    EncodingDetector() noexcept;

    base::EncodingType encoding_type;
};


/************************************
 * Inline Implementation
 ************************************/

 /**
 * get EncodingDetector(singleton design pattern).
 * @return EncodingDetector pointer. (using pointer instead of reference to avoid instance copy when using. )
 */
inline 
EncodingDetector* EncodingDetector::get_detector() noexcept
{
    static EncodingDetector singleton_instance;
    return &singleton_instance; 
}

/**
 * get console encoding (just according to compiling platform and priori knowledge).
 * @return priori knowledge about console encoding.
 */
inline 
base::EncodingType EncodingDetector::get_console_encoding() noexcept
{
    base::EncodingType type_from_priori = base::EncodingType::UTF8;
#ifdef _WIN32
    type_from_priori = base::EncodingType::GB18030;
#endif
    return type_from_priori;
}

 /**
 * constructor, just according to the platform.
 * WINDOWS -> GB18030, others -> UTF8
 */
#ifdef _WIN32
inline
EncodingDetector::EncodingDetector() noexcept
    : encoding_type(base::EncodingType::GB18030)
{}
#else
inline
EncodingDetector::EncodingDetector() noexcept
    : encoding_type(base::EncodingType::UTF8)
{}
#endif

/**
 * set encoding type.
 * @param encoding_type given encoding type.
 */
inline 
void EncodingDetector::set_encoding(base::EncodingType encoding_type) noexcept
{
    this->encoding_type = encoding_type;
}

inline
void EncodingDetector::set_encoding(std::string &encoding_name) noexcept
{
    EncodingType encoding_type = base::str2encoding_type(encoding_name);
    if( encoding_type != EncodingType::UNSUPPORT ){ set_encoding(encoding_type); }
    else
    {
        std::cerr << "un-supported encoding type: " << encoding_name << ". set to UTF8\n";
        set_encoding(base::EncodingType::UTF8);
    }
}

/**
 * get encoding type.
 * @return encoding type
 */
inline
base::EncodingType EncodingDetector::get_encoding() const noexcept
{
    return encoding_type;
}

/**
 * detect encoding from raw bytes.
 * TODO.
 */
inline
base::EncodingType EncodingDetector::detect_and_set_encoding(const std::string &bytes) noexcept
{
    // TODO !
    base::EncodingType detected_type_from_bytes = base::EncodingType::UTF8;
    set_encoding(detected_type_from_bytes);
    return detected_type_from_bytes;
}

/**
 * detect encoding from input file.
 * we'll just read limited bytes and call [ EncodingType (*)(const std::string &bytes) ].
 * TODO.
 */
inline
base::EncodingType EncodingDetector::detect_and_set_encoding(std::ifstream &f) noexcept
{
    // TODO !
    set_encoding(base::EncodingType::UTF8);
    return base::EncodingType::UTF8;
}

/**
 * detect encoding type from stdin.
 * we just according to the priori knowledge.
 * WINDOWS -> GB18030, OTHERS -> UTF8.
 */
inline 
base::EncodingType EncodingDetector::detect_and_set_encoding(std::istream &stdiof) noexcept
{
    base::EncodingType type_from_priori = get_console_encoding();
    set_encoding(type_from_priori);
    return type_from_priori;
}

/**
 * detect encoding from file stream explicitly.
 * when polymorphism, we can't use the completely-match rule to using the overload function for file stream.
 * eg:
 *   void f(std::istream &f){  detect_and_set_encoding(f); } -> we lose the original type, so un-expected things happen
 */
inline
base::EncodingType 
EncodingDetector::detect_and_set_encoding_from_fstream(std::ifstream &f) noexcept
{
    return detect_and_set_encoding(f);
}

} // end of namespace detector
using detector::EncodingDetector;
} // end of namespace charcode
} // end of namespace slnn


#endif