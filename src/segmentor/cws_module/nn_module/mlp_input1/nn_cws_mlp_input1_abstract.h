#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_ABSTRACT_H_
#include <functional>
#include "utils/typedeclaration.h"
#include "segmentor/cws_module/nn_module/nn_common_interface_cnn_impl.h"
#include "segmentor/cws_module/cws_feature_layer.h"
namespace slnn{
namespace segmentor{
namespace nn_module{

class NnSegmentorInput1Abstract : public NeuralNetworkCommonInterfaceCnnImpl
{
public:
    cnn::expr::Expression build_training_graph(const std::vector<Index> &charseq, const std::vector<Index> &tagseq);
    std::vector<Index> predict(const std::vector<Index> &charseq);
protected:
    std::shared_ptr<Index2ExprLayer> word_expr_layer;
    std::shared_ptr<WindowExprGenerateLayer> window_expr_generate_layer;
    std::shared_ptr<MLPHiddenLayer> mlp_hidden_layer;
    std::shared_ptr<BareOutputBase> output_layer;
};

std::function<cnn::expr::Expression(const cnn::expr::Expression &)> get_nonlinear_function_from_name(const std::string &name)
{
    std::string lower_name(name);
    for( char &c : lower_name ){ c = ::tolower(c); }
    if( lower_name == "relu" || lower_name == "rectify" ){ return &cnn::expr::rectify; }
    else if( lower_name == "sigmoid" || lower_name == "softmax" ){ return &cnn::expr::softmax; } // a bit strange...
    else if( lower_name == "tanh" ){ return &cnn::expr::tanh; }
    else
    {
        std::ostringstream oss;
        oss << "not supported non-linear funtion: " << name << "\n"  
            <<"Exit!\n";
        throw std::invalid_argument(oss.str());
    }
}


} // end of namespace nn_module
} // end of namespace segmentor
} // end of namespace slnn




#endif