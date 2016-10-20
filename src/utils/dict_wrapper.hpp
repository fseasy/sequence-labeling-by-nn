#ifndef DICT_WRAPPER_HPP_INCLUDED_
#define DICT_WRAPPER_HPP_INCLUDED_

#include "typedeclaration.h"
#include "dynet/dict.h"

#include <functional>
#include <algorithm>
#include <vector>

/**************************
 * DictWrapper
 * for record words frequency . 
 */
namespace slnn {
    struct DictWrapper
    {
        DictWrapper(dynet::Dict &d) : rd(d), freq_threshold(1), prob_threshold(0.2f), prob_rand(std::bind(std::uniform_real_distribution<float>(0, 1), *(dynet::rndeng) ))
        {
            freq_records.reserve(0xFFFF); // 60K space
        }

        inline int convert(const std::string& word)
        {
            Index word_idx = rd.convert(word);
            if (!rd.is_frozen())
            {
                if (static_cast<unsigned>(word_idx) + 1U > freq_records.size())
                {
                    // new words has been pushed to the dict !
                    freq_records.push_back(1); // add word frequency record
                }
                else ++freq_records[word_idx]; // update word_frequency record
            }
            return word_idx;
        }
        void set_unk(const std::string& word)
        {
            rd.set_unk(word);
            UNK = rd.convert(word);
        }
        void freeze() { rd.freeze(); }
        bool is_frozen() { return rd.is_frozen(); }
        int unk_replace_probability(Index word_idx)
        {
            if (word_idx == UNK) return UNK; // UNK is not in freq_records
            assert(static_cast<unsigned>(word_idx) < freq_records.size());
            if (freq_records[word_idx] <= freq_threshold && prob_rand() <= prob_threshold) return UNK;
            return word_idx;
        }
        void set_threshold(int freq_threshold, float prob_threshold)
        {
            this->freq_threshold = freq_threshold;
            this->prob_threshold = prob_threshold;
        }
        dynet::Dict &rd;
        std::vector<int> freq_records;
        Index UNK;
        int freq_threshold;
        float prob_threshold;
        std::function<float()> prob_rand;
    };
} // End of namespace slnn

#endif