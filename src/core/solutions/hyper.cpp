#include <core/solutions/hyper.h>
#include <core/generators/constant.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

hyper* hyper::make(generator* generator, algorithm* base, unsigned int size)
{
  auto* result = new hyper(generator, base, size);

  result->init();

  return result;
}

algorithm* hyper::get_algorithm() const
{
  return _base;
}

void hyper::set_do_optimize(std::function<void(algorithm*)> do_optimize)
{
  _do_optimize = do_optimize;
}

hyper* hyper::clone()
{
  auto* result = hyper::make(get_generator(), get_algorithm(), size());

  result->_fitness = fitness();
  result->set_modified(false);
  result->set_do_optimize(_do_optimize);

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
  set_do_optimize(_do_optimize);
}

float hyper::calculate_fitness()
{
  algorithm* base = get_algorithm();

  solution::calculate_fitness();

  base->set_params(size(), get_params());
  base->reset();

  _do_optimize(base);

  return base->get_best()->fitness();
}

hyper::hyper(generator* generator, algorithm* base, unsigned int size)
: solution(generator, size)
{
  _base = base;
  _do_optimize = [](algorithm* base)
  {
    base->optimize();
  };
}

hyper::~hyper()
{
}
 // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
