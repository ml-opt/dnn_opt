#include <core/solutions/hyper.h>
#include <core/generators/constant.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

hyper* hyper::make(generator* generator, algorithm* algorithm, unsigned int size)
{
  auto* result = new hyper(generator, algorithm, size);

  result->init();

  return result;
}

algorithm* hyper::get_algorithm() const
{
  return _algorithm;
}

int hyper::get_iteration_count() const
{
  return _iteration_count;
}

void hyper::set_iteration_count(int iteration_count)
{
  _iteration_count = iteration_count;
}

hyper* hyper::clone()
{
  auto* result = hyper::make(get_generator(), _algorithm, size());

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

float hyper::calculate_fitness()
{
  _algorithm->reset();
  _algorithm->set_params(size(), get_params());
  _algorithm->optimize(_iteration_count);

  return _algorithm->get_best()->fitness();
}

hyper::hyper(generator* generator, algorithm* algorithm, unsigned int size)
: solution(generator, size)
{
  _algorithm = algorithm;
  _iteration_count = 10;
}

hyper::~hyper()
{
}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
