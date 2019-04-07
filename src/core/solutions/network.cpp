#include <algorithm>
#include <cassert>
#include <core/solutions/network.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

float* network::CURRENT_OUT = 0;
float* network::PRIOR_OUT = 0;

network* network::make(generator* generator, reader* reader, error* error)
{
  auto* result = new network(generator, reader, error);

  result->init();

  return result;
}

network* network::clone()
{
  linked* nn = new linked(this);

  for(auto &l : _layers)
  {
    nn->add_layer(l->clone());
  }

  nn->init();
  nn->_fitness = fitness();
  nn->_evaluations = get_evaluations();
  nn->set_modified(false);

  std::copy_n(get_params(), size(), nn->get_params());

  return nn;
}

bool network::assignable(const solution* s) const
{
  /*
   * Warning: Incomplete method implementation.
   * Check also that contains the same layered structure.
   */

  return size() == s->size();
}

void network::add_layer(std::initializer_list<layer*> layers)
{
  for(layer* layer : layers)
  {
    add_layer(layer);
  }

  init();
}

network* network::add_layer(layer* layer)
{
  /* check input and output dimension */

  if (_layers.empty() == false)
  {
    assert(_layers.back()->get_out_dim() == layer->get_in_dim());
  }

  _size += layer->size();
  _max_out = std::max(_max_out, layer->get_out_dim());
  _layers.push_back(layer);

  return this;
}

void network::set_reader(reader* reader)
{
  if(_r->size() < reader->size())
  {
    delete[] CURRENT_OUT;
    delete[] PRIOR_OUT;

    CURRENT_OUT = new float[reader->size() * _max_out];
    PRIOR_OUT = new float[reader->size() * _max_out];
  }

  _r = reader;
  _modified = true;
}

float network::test(reader* validation_set)
{
  float result = 0;
  reader* current_reader = get_reader();

  set_reader(validation_set);
  result = calculate_fitness();
  set_reader(current_reader);

  return result;
}

float* network::predict(reader* validation_set)
{
  int n = validation_set->size() * validation_set->get_out_dim();
  float* result = new float[n];
  reader* current_reader = get_reader();

  set_reader(validation_set);
  std::copy_n(prop(), n, result);
  set_reader(current_reader);

  return result;
}

void network::init()
{
  reader* r = get_reader();

  delete[] CURRENT_OUT;
  delete[] PRIOR_OUT;

  CURRENT_OUT = new float[r->size() * _max_out];
  PRIOR_OUT = new float[r->size() * _max_out];

  solution::init();
}

reader* network::get_reader() const
{
  return _r;
}

error* network::get_error() const
{
  return _e;
}

float network::calculate_fitness()
{
  error* e = get_error();
  reader* r = get_reader();
  solution::calculate_fitness();

  e->ff(r->size(), r->get_out_dim(), prop(), r->out_data());

  return e->f();
}

const float* network::prop()
{
  reader* r = get_reader();
  float* params = get_params();

  /* propagate the signal in the first layer with input patterns */
  _layers.front()->prop(r->size(), r->in_data(), params, PRIOR_OUT);

  for(unsigned int i = 1; i < _layers.size(); i++)
  {
    /* move window to the parameters of this layer */
    params += _layers.at(i - 1)->size();

    /* propagate the signal through the layer */
    _layers.at(i)->prop(r->size(), PRIOR_OUT, params, CURRENT_OUT);

    /* swap the outs to use _current_out as _prior_out in next iteration */
    float* aux = PRIOR_OUT;
    PRIOR_OUT = CURRENT_OUT;
    CURRENT_OUT = aux;
  }

  return PRIOR_OUT;
}

network::network(generator* generator, reader* reader, error* error)
: solution(generator, 0)
{
  _r = reader;
  _e = error;
  _max_out = 0;
//  CURRENT_OUT = 0;
//  PRIOR_OUT = 0;
}

network::network(generator* generator)
: solution(generator, 0)
{
  _r = 0;
  _e = 0;
  _max_out = 0;
//  CURRENT_OUT = 0;
//  PRIOR_OUT = 0;
}

network::~network()
{
  for(auto* layer : _layers)
  {
    delete layer;
  }

//  delete[] CURRENT_OUT;
//  delete[] PRIOR_OUT;

//  CURRENT_OUT = 0;
//  PRIOR_OUT = 0;

  _layers.clear();
}

float network::linked::fitness()
{
  if(_base->get_reader() != network::get_reader())
  {
    set_reader(_base->get_reader());
  }

  return network::fitness();
}

reader* network::linked::get_reader() const
{
  return _base->get_reader();
}

void network::linked::set_reader(reader* reader)
{
  network::set_reader(reader);
  _base->set_reader(reader);
}

error* network::linked::get_error() const
{
  return _base->get_error();
}

network::linked::linked(network* base)
: solution(base->get_generator(), 0),
  network(base->get_generator())
{
  _base = base;
  _r = base->get_reader();
  _e = base->get_error();
}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
