#ifndef MISC_STR_UTILS_H_INCLUDED_
#define MISC_STR_UTILS_H_INCLUDED_
#include <unordered_set>
#include <string>
#include <vector>
#include <algorithm>

namespace misc {

/**
 * like boost::algorithm::split
 * *UnaryPredicate* should has the signature:
 *      bool predicate(const CharT&);
 */
template<typename CharT, typename UnaryPredicate>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s,
           UnaryPredicate predicate);
template<typename CharT>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s, 
           const std::basic_string<CharT>& char_candidate);
template<typename CharT>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s,
           const CharT* literals);

template<typename CharT, typename UnaryPredicate>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                               UnaryPredicate predicate);
template<typename CharT>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                               const std::basic_string<CharT>& char_candidate);
template<typename CharT>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                                            const CharT* literals);







/**
 * inline/template implementation
 */

template<typename CharT, typename UnaryPredicate>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s,
           UnaryPredicate predicate)
{
    using ST = std::basic_string<CharT>;
    using std::swap;
    std::vector<ST> tmp_result;
    auto iter = s.cbegin(),
         end_iter = s.cend();
    while (true)
    {
        /**
         * edge case: empty str -> push an empty str and exit.
         */
        auto find_iter = find_if(iter, end_iter, predicate);
        tmp_result.emplace_back(iter, find_iter);
        if (find_iter == end_iter) { break; }
        iter = ++find_iter; 
    }
    swap(tmp_result, split_result);
}


template<typename CharT>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s,
           const std::basic_string<CharT>& char_candidate)
{
    std::unordered_set<CharT> candidate_set(char_candidate.cbegin(),
                                            char_candidate.cend());
    auto predicate = [&candidate_set](const CharT& c) {
        return candidate_set.count(c) > 0U;
    };
    return split(split_result, s, predicate);
}

template<typename CharT>
void split(std::vector<std::basic_string<CharT>>& split_result,
           const std::basic_string<CharT>& s,
           const CharT* literals)
{
    return split(split_result, s, std::basic_string<CharT>(literals));
}


template<typename CharT, typename UnaryPredicate>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                                            UnaryPredicate predicate)
{
    std::vector<std::basic_string<CharT>> result;
    split(result, s, predicate);
    return result;
}

template<typename CharT>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                                            const std::basic_string<CharT>& char_candidate)
{
    std::unordered_set<CharT> candidate_set(char_candidate.cbegin(),
                                            char_candidate.cend());
    auto predicate = [&candidate_set](const CharT& c) {
        return candidate_set.count(c) > 0U;
    };
    return split(s, predicate);
}

template<typename CharT>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& s,
                                            const CharT* literals)
{
    return split(s, std::basic_string<CharT>(literals));
}


} // end of namespace misc


#endif