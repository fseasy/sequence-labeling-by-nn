#include "layers.h" 

using namespace cnn;
using namespace std;
namespace slnn {

// Bi-LSTM

BILSTMLayer::BILSTMLayer(Model *m , unsigned nr_lstm_stacked_layers, unsigned lstm_x_dim, unsigned lstm_h_dim ,
                         cnn::real default_dropout_rate)
    : l2r_builder(new LSTMBuilder(nr_lstm_stacked_layers , lstm_x_dim , lstm_h_dim , m)) ,
    r2l_builder(new LSTMBuilder(nr_lstm_stacked_layers , lstm_x_dim , lstm_h_dim , m)) ,
    SOS(m->add_parameters({lstm_x_dim})) ,
    EOS(m->add_parameters({lstm_x_dim})) ,
    default_dropout_rate(default_dropout_rate)
{}

BILSTMLayer::~BILSTMLayer()
{ 
    if (l2r_builder) delete l2r_builder;
    if (r2l_builder) delete r2l_builder;
}

// DenseLayer

DenseLayer::DenseLayer(Model *m , unsigned input_dim , unsigned output_dim)
    :w(m->add_parameters({output_dim , input_dim})) ,
    b(m->add_parameters({output_dim}))
{}

DenseLayer::~DenseLayer(){}

// Merge 2 Layer

Merge2Layer::Merge2Layer(Model *m, unsigned input1_dim, unsigned input2_dim,unsigned output_dim)
    :w1(m->add_parameters({ output_dim , input1_dim })),
    w2(m->add_parameters({ output_dim , input2_dim })),
    b(m->add_parameters({ output_dim}))
{}

Merge2Layer::~Merge2Layer() {}


// Merge 3 Layer

Merge3Layer::Merge3Layer(Model *m ,unsigned input1_dim , unsigned input2_dim , unsigned input3_dim , unsigned output_dim )
    :w1(m->add_parameters({output_dim , input1_dim})) ,
    w2(m->add_parameters({output_dim , input2_dim})) ,
    w3(m->add_parameters({output_dim , input3_dim})) ,
    b(m->add_parameters({output_dim}))
{}

Merge3Layer::~Merge3Layer(){}



} // end namespace slnn