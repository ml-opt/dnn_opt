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

  for(auto &l : _layers)
  {
    nn->add_layer(l->clone());
  }

  nn->init();

  nn->_fitness = fitness();
  nn->_evaluations = get_evaluations();
  nn->_modified = false;

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
  return _copt_reader;
}

void network::set_reader(core::reader* reader)
{
  core::solutions::network::set_reader(reader);
  _copt_reader = dynamic_cast<copt::reader*>(reader);
}

error* network::get_error() const
{
  return _copt_error;
}

reader* network::linked::get_reader() const
{
  return dynamic_cast<reader*>(_base->get_reader());
}

void network::linked::set_reader(core::reader* reader)
{
  network::set_reader(reader);
  _base->set_reader(reader);
}

error* network::linked::get_error() const
{
  return dynamic_cast<error*>(_base->get_error());
}

network::network(generator* generator, reader* reader, error* error)
: solution(generator, 0),
  core::solution(generator, 0),
  core::solutions::network(generator, reader, error)
{
  _copt_reader = reader;
  _copt_error = error;
}

network::network(generator* generator)
: solution(generator, 0),
  core::solution(generator, 0),
  core::solutions::network(generator)
{
  _copt_reader = 0;
  _copt_error = 0;
}

network::~network()
{

}

network::linked::linked(network* base)
: solution(base->get_generator(), 0),
  network(base->get_generator()),
  core::solution(base->get_generator(), 0),
  core::solutions::network(base->get_generator()),
  core::solutions::network::linked(base)
{

}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
