#ifndef SYMNN_MODEL_H_INCLUDED_
#define SYMNN_MODEL_H_INCLUDED_

#include <cstddef>
#include <vector>
#include <unordered_set>

#include "symnn/weight_decay.h"
#include "symnn/tensor.h"
#include "symnn/dim.h"
#include "symnn/type.h"
namespace symnn {

class ParameterInit;
class Model;

class ParameterStorageBase
{
    friend class model;

public:
    virtual void scale_parameter(float a) = 0;

    virtual void scale_gradient(float a) = 0;

    virtual void zero() = 0;

    virtual void squared_l2norm(float* sqnorm) const = 0;

    virtual void g_squared_l2norm(float* sqnorm) const = 0;

    virtual std::size_t size() const = 0;

    virtual ~ParameterStorageBase();
};

class ParameterStorage : public ParameterStorageBase
{
    friend class Model;
public:
    void scale_parameter(float a) override;
    void scale_gradient(float a) override;
    void squared_l2norm(float* sqnorm) const override;
    void g_squared_l2norm(float* sqnorm) const override;
    std::size_t size() const override;

    void copy(const ParameterStorage& val);

    void accumulate_grad(const Tensor& g);

    void clear();

    void clip(float left, float right);

    const Dim& get_dimension() const { return dim; }
    Tensor* get_value() { return &value; }
private:
    ParameterStorage() {}
    explicit ParameterStorage(const Dim& d, float minmax);
    explicit ParameterStorage(const Dim& d, const ParameterInit& init);
private:
    Dim dim;
    Tensor value;
    Tensor g;
};

class LookupParameterStorage : public ParameterStorageBase
{
    friend class Model;
public:
    void scale_parameter(float a) override;
    void scale_gradient(float a) override;
    void squared_l2norm(float* sqnorm) const override;
    void g_squared_l2norm(float* sqnorm) const override;
    std::size_t size() const override;

    void initialize(unsigned index, const std::vector<real_t> val);

    void copy(const LookupParameterStorage& val);

    void accumulate_grad(unsigned index, const Tensor& g);

    void clear();
    void initialize_lookups();

    const Dim& get_dimension() const { return dim; }
    std::vector<Tensor>* get_values() { return &values; }

    const Dim& get_all_dimension() const { return all_dim; }
private:
    LookupParameterStorage() :all_updated(false) {};
    LookupParameterStorage(unsigned n, const Dim& d);
    LookupParameterStorage(unsigned n, const Dim& d, const ParameterInit& init);

private:
    Dim all_dim;
    Tensor all_values;
    Tensor all_grads;

    Dim dim;
    std::vector<Tensor> values;
    std::vector<Tensor> grads;
    std::unordered_set<unsigned> non_zero_grads;
    bool all_updated;
};


class Parameter
{
    friend class Model;
public:
    ParameterStorage* get() const;
    void zero();

    const Dim& get_dimension() { return get()->get_dimension(); }
    Tensor* get_value() { return get()->get_value(); }
    void set_updated(bool b);
    void scale(float s) { get()->scale_parameter(s); }
    void scale_gradient(float s) { get()->scale_gradient(s); }
    bool is_updated();

    void clip_inplace(float left, float right);
public:
    Parameter();
    Parameter(Model *pm, unsigned index);

private:
    Model* pm;
    unsigned index;
};

class LookupParameter {
    friend class Model;
public:
    LookupParameter();
    LookupParameter(Model* pm, unsigned index);

public:
    LookupParameterStorage* get() const;

    void initialize(unsigned index, const std::vector<real_t>& val) const;

    void zero();

    const Dim& get_dimension() const { return get()->get_dimension(); }
    std::vector<Tensor>* get_values() { return get()->get_values(); }

    const Dim& get_all_dimension() const { return get()->get_all_dimension(); }

    void scale(float s) { get()->scale_parameter(s); }
    void scale_gradient(float s) { get()->scale_gradient(s); }

    void set_updated(bool b);
    bool is_updated();
private:
    Model* pm;
    unsigned index;

};

class ParameterInit
{
public:
    ParameterInit() {};
    virtual ~ParameterInit() {}
    virtual void initialize_params(Tensor& value) const = 0;
};

class ParameterInitNormal : public ParameterInit
{
public:
    ParameterInitNormal(real_t m = 0.f, real_t v = 1.0f) : mean(m), var(v) {}
    virtual void initialize_params(Tensor& value) const override;
private:
    real_t mean;
    real_t var;
};

class ParameterInitUniform : public ParameterInit 
{
public:
    ParameterInitUniform(real_t scale) :left(-scale), right(scale)
    {
        if (std::abs(scale - 0.f) < 1e-6f)
        {
            throw std::domain_error("Scale of unifrom can't be 0");
        }
    }
    
    ParameterInitUniform(real_t l, real_t r) : left(l), right(r)
    {
        if (std::abs(l - r) < 1e-6f)
        {
            throw std::domain_error("Empty interval for uniform");
        }
    }

    virtual void initialize_params(Tensor& value) const override;
private:
    real_t left;
    real_t right;

};

class ParameterInitConst : public ParameterInit 
{
public:
    ParameterInitConst(real_t c) : cnst(c) {}
    virtual void initialize_params(Tensor& value) const override;

private:
    real_t cnst;
};

class ParameterInitIdentity : public ParameterInit
{
public:
    ParameterInitIdentity() {}
    virtual void initialize_params(Tensor& value) const override;
};

class ParameterInitGlorot : public ParameterInit
{
public:
    ParameterInitGlorot(bool is_lookup = false, real_t gain = 1.f) : lookup(is_lookup), gain(gain) {}
    virtual void initialize_params(Tensor& values) const override;
private:
    bool lookup;
    real_t gain;
};

class ParameterInitSaxe : public ParameterInit
{
public:
    ParameterInitSaxe(real_t gain) : gain(gain) {}

    virtual void initialize_params(Tensor& value) const override;

private:
    real_t gain;
};

class ParameterInitFromFile : public ParameterInit
{
public:
    ParameterInitFromFile(std::string f) : filename(f) {}
    virtual void initialize_params(Tensor& value) const override;
private:
    std::string filename;
};

class ParameterInitFromVector : public ParameterInit
{
public:
    ParameterInitFromVector(const std::vector<real_t>& v) : vec(v) {}
    virtual void initialize_params(Tensor &value) const override;
private:
    std::vector<real_t> vec;
};


class Model
{
public:
    Model();
    ~Model();

public:
    real_t gradient_l2_norm() const;
    void reset_gradient();

    Parameter add_parameter(const Dim& d, real_t scale = 0.f);
    Parameter add_parameter(const Dim& d, const ParameterInit& init);

    LookupParameter add_lookup_parameter(unsigned n, const Dim& d);
    LookupParameter add_lookup_parameter(unsigned n, const Dim& d, 
                                         const ParameterInit& init);

    void project_weights(real_t radius = 1.f);

    void set_weight_decay_lambda(float lambda);

    std::size_t parameter_count() const;
    std::size_t updated_parameter_count() const;

protected:
    const std::vector<ParameterStorage*>& parameter_list() const { return params; }
    const std::vector<LookupParameterStorage*>& lookup_parameter_list() const { return lookup_params; }
    const std::vector<unsigned>& updated_paramerter_list() const { return updated_params; }
    const std::vector<unsigned>& updated_lookup_parameter_list() const { return updated_lookup_params; }

private:
    L2WeightDecay weight_decay;
    std::vector<ParameterStorageBase*> all_params;
    std::vector<ParameterStorage*> params;
    std::vector<LookupParameterStorage*> lookup_params;

    std::vector<unsigned> updated_params;
    std::vector<unsigned> updated_lookup_params;

    mutable float* gradient_norm_scratch;
};

} // end of namespace symnn

#endif