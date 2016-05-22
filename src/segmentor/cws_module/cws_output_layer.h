#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_OUTPUT_LAYER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_OUTPUT_LAYER_H_

#include "modelmodule/hyper_layers.h"
#include "segmentor/cws_module/cws_tagging_system.h"

namespace slnn{

struct CWSSimpleOutput : SimpleOutput
{
    // add tagging system viriable
    // and add comstrained decoding when predict 
    CWSTaggingSystem &tag_sys ;
    CWSSimpleOutput(cnn::Model *m,
                    unsigned input_dim1, unsigned input_dim2,
                    unsigned hidden_dim, unsigned output_dim,
                    CWSTaggingSystem &tag_sys,
                    NonLinearFunc *nonlinear_func = &cnn::expr::rectify) ;
    void build_output(const std::vector<cnn::expr::Expression> &expr_cont1,
                      const std::vector<cnn::expr::Expression> &expr_cont2,
                      IndexSeq &pred_out_seq);

protected :
    Index select_pred_tag_in_constrain(std::vector<cnn::real> &dist, int pos , Index pre_tag) ;

};


} // end of namespace slnn
#endif 