#ifndef SLNN_TRIVIAL_CHARCODE_DETECTOR_H_
#define SLNN_TRIVIAL_CHARCODE_DETECTOR_H_
#include <fstream>
#include <iostream>
#include <algorithm>
#include "charcode_base.hpp"

namespace slnn{
namespace charcode{
namespace detector{

class EncodingDetector{
public:
    EncodingDetector& get_detector() noexcept;
    void set_encoding(base::EncodingType) noexcept;
    void set_encoding(std::string &encoding_name) noexcept;
    base::EncodingType get_encoding() const noexcept;
    base::EncodingType detect_and_set_encoding(const std::string &bytes) noexcept;
    base::EncodingType detect_and_set_encoding(std::ifstream &f) noexcept; // reject iostream
    base::EncodingType detect_and_set_encoding(std::istream &stdinf) noexcept;
private:
    EncodingDetector();
    base::EncodingType encoding_type;
};


/************************************
 * Inline Implementation
 ************************************/
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
 * get EncodingDetector(singleton design pattern).
 * @return EncodingDetector reference.
 */
inline 
EncodingDetector& EncodingDetector::get_detector() noexcept
{
    static EncodingDetector singleton_instance;
    return singleton_instance;
}

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
    std::string upper_name;
    // Atention.
    // WHY static_cast ?
    // toupper has multi definition, one in global namespace from <cctype> : int toupper(int)
    // one from namespace std : template <class charT> charT toupper(charT c, const locale& loc)
    // in fact, we haven't using namespace std, so it is ok for just using toupper wihout any namespae prefix.
    // but I think it is also misleading. SO it is better write it directly!
    // For more information, see : http://stackoverflow.com/questions/7131858/stdtransform-and-toupper-no-matching-function
    // Actually, using lambda expression may be more recommending way( it is the sample from cppreference ).
    std::transform(encoding_name.cbegin(), encoding_name.cend(), std::back_inserter(upper_name),
        static_cast<int(*)(int)>(&std::toupper));
    if( uppper_name == "UTF8" ){ set_encoding(base::EncodingType::UTF8); }
    else if( uppper_name == "GB18030"
             uppper_name == "GB2312"){ set_encoding(base::EncodingType::GB18030); }
    else
    {
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
base::EncodingType EncodingDetector::detect_and_set_encoding(std::ifstream &stdiof) noexcept
{
    base::EncodingType type_from_priori = base::EncodingType::UTF8;
#ifdef _WIN32
    type_from_priori = base::EncodingType::GB18030;
#endif
    set_encoding(type_from_priori);
    return type_from_priori;
}

} // end of namespace detector
} // end of namespace charcode
} // end of namespace slnn


#endif