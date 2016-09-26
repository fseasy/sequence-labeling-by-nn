#include "cws_base_model.h"

using namespace std;
using namespace slnn;

const string CWSBaseModel::UNK_STR = "unk_str";
const unsigned CWSBaseModel::SentMaxLen = 256;

CWSBaseModel::CWSBaseModel(unsigned seed) noexcept
    :char_dict(seed)
{}