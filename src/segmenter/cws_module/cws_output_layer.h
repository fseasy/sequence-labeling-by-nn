#ifndef SLNN_SEGMENTER_CWS_MODULE_CWS_OUTPUT_LAYER_H_
#define SLNN_SEGMENTER_CWS_MODULE_CWS_OUTPUT_LAYER_H_

#include "modelmodule/hyper_layers.h"
#include "segmenter/cws_module/cws_tagging_system.h"

namespace slnn{

struct CWSSimpleOutput : SimpleOutput
{
    // add tagging system viriable
    // and add comstrained decoding when predict 
    CWSTaggingSystem &tag_sys ;
    CWSSimpleOutput(dynet::Model *m,
        unsigned input_dim1, unsigned input_dim2,
        unsigned hidden_dim, unsigned output_dim,
        CWSTaggingSystem &tag_sys,
        dynet::real dropout_rate = 0.f,
        NonLinearFunc *nonlinear_func = &dynet::expr::rectify) ;
    void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
                      const std::vector<dynet::expr::Expression> &expr_cont2,
                      IndexSeq &pred_out_seq);

protected :
    Index select_pred_tag_in_constrain(std::vector<dynet::real> &dist, size_t pos , Index pre_tag) ;

};

struct CWSPretagOutput : PretagOutput
{
    CWSTaggingSystem &tag_sys ;
    CWSPretagOutput(dynet::Model *m,
        unsigned tag_embedding_dim,
        unsigned input_dim1, unsigned input_dim2,
        unsigned hidden_dim, unsigned output_dim,
        CWSTaggingSystem &tag_sys,
        dynet::real dropout_rate = 0.f,
        NonLinearFunc *nonlinear_fun = &dynet::expr::rectify);
    void build_output(const std::vector<dynet::expr::Expression> &expr_1,
                      const std::vector<dynet::expr::Expression> &expr_2,
                      IndexSeq &pred_out_seq) ;
protected :
    Index select_pred_tag_in_constrain(std::vector<dynet::real> &dist, size_t pos , Index pre_tag) ;
};

struct CWSCRFOutput : CRFOutput
{
    CWSTaggingSystem &tag_sys ;
    CWSCRFOutput(dynet::Model *m,
                 unsigned tag_embedding_dim, unsigned input_dim1, unsigned input_dim2,
                 unsigned hidden_dim,
                 unsigned tag_num,
                 dynet::real dropout_rate,
                 CWSTaggingSystem &tag_sys,
                 NonLinearFunc *nonlinear_func = &dynet::expr::rectify) ;

    dynet::expr::Expression 
    build_output_loss(const std::vector<dynet::expr::Expression> &expr_cont1,
                        const std::vector<dynet::expr::Expression> &expr_cont2,
                        const IndexSeq &gold_seq) ;

    void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
                      const std::vector<dynet::expr::Expression> &expr_cont2,
                      IndexSeq &pred_seq) ;

};


struct CWSSimpleOutputWithFeature : SimpleOutputWithFeature
{
    CWSSimpleOutputWithFeature(dynet::Model *m, unsigned input_dim1, unsigned input_dim2, unsigned feature_dim,
        unsigned hidden_dim, unsigned output_dim,
        dynet::real dropout_rate=0.f, NonLinearFunc *nonlinear_func=&dynet::expr::rectify);
    virtual void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        const std::vector<dynet::expr::Expression> &feature_expr_cont,
        IndexSeq &pred_out_seq);
};

// After 0628, we abandon CWSTaggingSystem instance .
// Have to write another one . 

struct CWSSimpleOutputNew : SimpleOutput
{
    CWSSimpleOutputNew(dynet::Model *m,
        unsigned input_dim1, unsigned input_dim2,
        unsigned hidden_dim, unsigned output_dim,
        dynet::real dropout_rate=0.f,
        NonLinearFunc *nonlinear_func = &dynet::expr::rectify) ;
    void build_output(const std::vector<dynet::expr::Expression> &expr_cont1,
        const std::vector<dynet::expr::Expression> &expr_cont2,
        IndexSeq &pred_out_seq) override ;

};

} // end of namespace slnn
#endif 