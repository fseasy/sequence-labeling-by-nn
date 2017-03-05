#include "model_utility.h"

namespace slnn{
namespace ner{
namespace utility{

void annotated_instance_list2feat_list_and_ner_tag_list(
    const std::vector<AnnotatedInstance>& instance_list, 
    const TokenDict& token_dict,
    std::vector<InstanceFeature>& feat_list, 
    std::vector<NerTagIndexSeq>& ner_tag_list)
{
    std::size_t sz = instance_list.size();
    feat_list.resize(sz);
    ner_tag_list.resize(sz);
    std::transform(instance_list.begin(), instance_list.end(), 
        feat_list.begin(), 
        [&token_dict](const AnnotatedInstance& instance) -> InstanceFeature{
            token_module::instance2feature(instance, token_dict);
    });
    std::transform(instance_list.begin(), instance_list.end(),
        ner_tag_list.begin(),
        [&token_dict](const AnnotatedInstance& instance){
        token_module::ner_seq2ner_index_seq(instance.ner_tag_seq, token_dict);
    });
}

void unannotated_instance_list2feat_list(
    const std::vector<UnannotatedInstance>& u_instance_list,
    const TokenDict& token_dict,
    std::vector<InstanceFeature>& feat_list)
{
    std::size_t sz = u_instance_list.size();
    feat_list.resize(sz);
    std::transform(u_instance_list.begin(), u_instance_list.end(), 
        feat_list.begin(), 
        [&token_dict](const UnannotatedInstance& instance) -> InstanceFeature{
        token_module::instance2feature(instance, token_dict);
    });
}

void save(std::ostream& os, const TokenDict& dict, const StructureParam& param,
    NnMlp& m)
{
    boost::archive::text_oarchive to(os);
    to << dict;
    to << param;
    unsigned seed = m.get_rng_seed(); // boost can't save r-value for unsigned??
    to << seed; 
    m.reset2stashed_model();
    to << *m.get_dynet_model();
}

void load(std::istream& is, int argc, char** argv,
    std::shared_ptr<TokenDict>& pdict, 
    std::shared_ptr<StructureParam>& pparam,
    std::shared_ptr<NnMlp>& pm)
{
    boost::archive::text_iarchive ti(is);
    pdict.reset(new TokenDict());
    ti >> *pdict;
    pparam.reset(new StructureParam());
    ti >> *pparam;
    unsigned seed;
    ti >> seed;
    pm.reset(new NnMlp(argc, argv, seed, pparam));
    pm->build_model_structure();
    ti >> *(pm->get_dynet_model());

}

} // end of namespace utility
} // end of namespace ner
} // end of namespace slnn