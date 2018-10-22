#include <algorithm>
#include <cassert>
#include <core/solutions/network.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

network* network::make(generator* generator, reader* reader, error* error)
{
  auto* result = new network(generator, reader, error);

  result->init();

  return result;
}

network* network::clone()
{
  linked* nn = new linked(this);

  nn->_fitness = fitness();
  nn->_evaluations = get_evaluations();
  nn->_modified = false;

  for(auto &l : _layers)
  {
    nn->add_layer(l->clone());
  }

  nn->init();
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
  _r = reader;
  _modified = true;

  delete[] _current_out;
  delete[] _prior_out;

  _current_out = new float[_r->size() * _max_out];
  _prior_out = new float[_r->size() * _max_out];
}

float network::test(reader* validation_set)
{
  float result = 0;
  reader* current_reader = _r;

  set_reader(validation_set);
  result = calculate_fitness();
  set_reader(current_reader);

  return result;
}

void network::init()
{
  delete[] _current_out;
  delete[] _prior_out;

  _current_out = new float[_r->size() * _max_out];
  _prior_out = new float[_r->size() * _max_out];

  solution::init();
}

reader* network::get_reader()
{
  return _r;
}

error* network::get_error() const
{
  return _e;
}

float network::calculate_fitness()
{
  solution::calculate_fitness();

  _e->ff(_r->size(), _r->get_out_dim(), prop(), _r->out_data());

  return _e->f();
}

const float* network::prop()
{
  float* params = get_params();

  /* propagate the signal in the first layer with input patterns */
  _layers.front()->prop(_r->size(), _r->in_data(), params, _prior_out);

  for(unsigned int i = 1; i < _layers.size(); i++)
  {
    /* move window to the parameters of this layer */
    params += _layers.at(i - 1)->size();

    /* propagate the signal through the layer */
    _layers.at(i)->prop(_r->size(), _prior_out, params, _current_out);

    /* swap the outs to use _current_out as _prior_out in next iteration*/
    float* aux = _prior_out;
    _prior_out = _current_out;
    _current_out = aux;
  }

  return _prior_out;
}

network::network(generator* generator, reader* reader, error* error)
: solution(generator, 0)
{
  _r = reader;
  _e = error;
  _max_out = 0;
  _current_out = 0;
  _prior_out = 0;
}

network::~network()
{
  for(auto* layer : _layers)
  {
    delete layer;
  }

  delete[] _current_out;
  delete[] _prior_out;

  _layers.clear();
}

reader* network::linked::get_reader()
{
  if(_source->get_reader() != network::get_reader())
  {
    network::set_reader(_source->get_reader());
  }

  return network::get_reader();
}

network::linked::linked(network* source)
: solution(source->get_generator(), 0),
  network(source->get_generator(), source->get_reader(), source->get_error())
{
  _source = source;
}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
