#include <copt/solutions/hyper.h>
#include <copt/generators/constant.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{

hyper* hyper::make(generator* generator, algorithm* algorithm, unsigned int size)
{
  auto* result = new hyper(generator, algorithm, size);

  result->init();

  return result;
}

hyper* hyper::clone()
{
  auto* result = hyper::make(get_generator(), get_algorithm(), size());

  result->_fitness = fitness();
  result->set_modified(false);
  result->set_iteration_count(get_iteration_count());

  result->_evaluations = get_evaluations();

  std::copy_n(get_params(), size(), result->get_params());

  return result;
}

bool hyper::assignable(const solution* s) const
{
  return size() == s->size();
}

void hyper::assign(solution* s)
{
  /* TODO: check if assignable */

  auto* ss = dynamic_cast<hyper*>(s);

  solution::assign(ss);
  set_iteration_count(ss->get_iteration_count());
}

algorithm* hyper::get_algorithm() const
{
  return _copt_algorithm;
}

hyper::hyper(generator* generator, algorithm* algorithm, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::hyper(generator, algorithm, size)
{
  _copt_algorithm = algorithm;
}

hyper::~hyper()
{
}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
