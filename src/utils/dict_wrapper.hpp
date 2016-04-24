#include "typedeclaration.h"
#include "cnn/dict.h"

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
        DictWrapper(cnn::Dict &d) : rd(d), freq_threshold(1), prob_threshold(0.2f), prob_rand(std::bind(std::uniform_real_distribution<float>(0, 1), *(cnn::rndeng) ))
        {
            freq_records.reserve(0xFFFF); // 60K space
        }

        inline int Convert(const std::string& word)
        {
            Index word_idx = rd.Convert(word);
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
        void SetUnk(const std::string& word)
        {
            rd.SetUnk(word);
            UNK = rd.Convert(word);
        }
        void Freeze() { rd.Freeze(); }
        int ConvertProbability(Index word_idx)
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
        cnn::Dict &rd;
        std::vector<int> freq_records;
        Index UNK;
        int freq_threshold;
        float prob_threshold;
        function<float()> prob_rand;
    };
} // End of namespace slnn