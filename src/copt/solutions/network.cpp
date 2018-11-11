#include <algorithm>
#include <cassert>
#include <copt/solutions/network.h>

namespace dnn_opt
{
namespace copt
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

bool network::assignable(const dnn_opt::core::solution* s) const
{
  /*
   * Warning: Incomplete method implementation.
   * Check also that contains the same layered structure.
   */

  return size() == s->size();
}

reader* network::get_reader() const
{
  return dynamic_cast<reader*>(_r);
}

error* network::get_error() const
{
  return dynamic_cast<error*>(_e);
}

network::network(generator* generator, reader* reader, error* error)
: solution(generator, 0),
  core::solution(generator, 0),
  core::solutions::network(generator, reader, error)
{

}

network::~network()
{

}

network::linked::linked(network* source)
: solution(source->get_generator(), 0),
  network(source->get_generator(), source->get_reader(), source->get_error()),
  core::solution(source->get_generator(), 0),
  core::solutions::network(source->get_generator(), source->get_reader(), source->get_error()),
  core::solutions::network::linked(source)
{
  _source = source;
}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
