#include "hyper_input_layers.h"

namespace slnn{

/********** Input1 *********/

Input1::Input1(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim)
    :word_lookup_param(m->add_lookup_parameters(vocab_size, {embedding_dim}))
{}

Input1::~Input1()
{}

/**********  Input1WithFeature  *********/

Input1WithFeature::Input1WithFeature(cnn::Model *m, unsigned vocab_size, unsigned embedding_dim,
    unsigned feature_embedding_dim, unsigned merge_out_dim,
    NonLinearFunc *nonlinear_func)
    :word_lookup_param(m->add_lookup_parameters(vocab_size, { embedding_dim })),
    m2_layer(m,embedding_dim, feature_embedding_dim, merge_out_dim),
    nonlinear_func(nonlinear_func)
{}

/********** Input2D ***********/

Input2D::Input2D(cnn::Model *m, unsigned vocab_size1, unsigned embedding_dim1,
    unsigned vocab_size2, unsigned embedding_dim2,
    unsigned mergeout_dim ,
    NonLinearFunc *nonlinear_func)
    : dynamic_lookup_param1(m->add_lookup_parameters(vocab_size1, {embedding_dim1})) ,
    dynamic_lookup_param2(m->add_lookup_parameters(vocab_size2 , {embedding_dim2})) ,
    m2_layer(m , embedding_dim1 , embedding_dim2 , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input2D::~Input2D() {}

/************* Input2 ***************/

Input2::Input2(cnn::Model *m, unsigned dynamic_vocab_size, unsigned dynamic_embedding_dim,
    unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
    unsigned mergeout_dim ,
    NonLinearFunc *nonlinear_func) 
    :dynamic_lookup_param(m->add_lookup_parameters(dynamic_vocab_size, {dynamic_embedding_dim})) ,
    fixed_lookup_param(m->add_lookup_parameters(fixed_vocab_size , {fixed_embedding_dim})) ,
    m2_layer(m , dynamic_embedding_dim , fixed_embedding_dim , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input2::~Input2() {};

/* Input2 with Feature */

Input2WithFeature::Input2WithFeature(cnn::Model *m, unsigned dynamic_vocab_size, unsigned dynamic_embedding_dim,
    unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
    unsigned feature_embedding_dim,
    unsigned mergeout_dim , NonLinearFunc *nonlinear_func)
    :dynamic_lookup_param(m->add_lookup_parameters(dynamic_vocab_size, {dynamic_embedding_dim})) ,
    fixed_lookup_param(m->add_lookup_parameters(fixed_vocab_size , {fixed_embedding_dim})) ,
    m3_layer(m,dynamic_embedding_dim, fixed_embedding_dim, feature_embedding_dim, mergeout_dim),
    nonlinear_func(nonlinear_func)
{}

Input2WithFeature::~Input2WithFeature(){};

/*********** Input3 *********/

Input3::Input3(cnn::Model *m, unsigned dynamic_vocab_size1, unsigned dynamic_embedding_dim1,
    unsigned dynamic_vocab_size2, unsigned dynamic_embedding_dim2,
    unsigned fixed_vocab_size, unsigned fixed_embedding_dim,
    unsigned mergeout_dim,
    NonLinearFunc *nonlinear_func)
    : dynamic_lookup_param1(m->add_lookup_parameters(dynamic_vocab_size1 , {dynamic_embedding_dim1})) ,
    dynamic_lookup_param2(m->add_lookup_parameters(dynamic_vocab_size2 , {dynamic_embedding_dim2})) ,
    fixed_lookup_param(m->add_lookup_parameters(fixed_vocab_size , {fixed_embedding_dim})) ,
    m3_layer(m , dynamic_embedding_dim1 , dynamic_embedding_dim2 , fixed_embedding_dim , mergeout_dim) ,
    nonlinear_func(nonlinear_func)
{}

Input3::~Input3(){}

/************ Bare Input1 *********/

BareInput1::BareInput1(cnn::Model *m, unsigned vocabulary_size, unsigned word_embedding_dim, unsigned nr_extra_feature_expr)
    :word_lookup_param(m->add_lookup_parameters(vocabulary_size, {word_embedding_dim})),
    nr_exprs(nr_extra_feature_expr+1),    exprs(nr_exprs) // pre-allocate memory
{}

/*  Another Bare input1 */
AnotherBareInput1::AnotherBareInput1(cnn::Model *m, unsigned vocabulary_size, unsigned word_embedding_dim)
    :word_lookup_param(m->add_lookup_parameters(vocabulary_size, {word_embedding_dim}))
{}

} // end of namespace slnn