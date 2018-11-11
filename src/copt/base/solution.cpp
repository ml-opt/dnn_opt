#include <algorithm>
#include <stdexcept>
#include <copt/base/solution.h>

namespace dnn_opt
{
namespace copt
{

solution* solution::make(generator* generator, unsigned int size)
{
  auto* result = new solution(generator, size);

  result->init();

  return result;
}

solution* solution::clone()
{
  auto* result = solution::make(dynamic_cast<generator*>(get_generator()), size());

  std::copy_n(get_params(), size(), result->get_params());

  result->_fitness = fitness();
  result->_evaluations = get_evaluations();
  result->_modified = false;

  return result;
}

bool solution::assignable(const solution* s) const
{
  return s->size() == size();
}

void solution::assign(solution* s)
{
  if(assignable(s) == false)
  {
    throw new std::invalid_argument("solution is not compatible");
  }

  std::copy_n(s->get_params(), size(), get_params());

  _fitness = s->fitness();
  _evaluations = s->get_evaluations();

  set_modified(false);
}

generator* solution::get_generator() const
{
  return _copt_generator;
}

solution::solution(generator* generator, unsigned int size)
: core::solution(generator, size)
{
  _copt_generator = generator;
}

} // namespace copt
} // namespace dnn_opt
