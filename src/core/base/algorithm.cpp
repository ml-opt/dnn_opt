#include <cmath>
#include <core/base/algorithm.h>

namespace dnn_opt
{
namespace core
{

void algorithm::optimize()
{
  _iterations += 1;
}

void algorithm::optimize(int count, std::function<bool()> on)
{
  bool on_opt = true;

  reset();

  for(int i = 0; i < count && on_opt; i++)
  {
    optimize();
    on_opt = on();
  }
}

void algorithm::optimize_idev(int count, float dev, std::function<bool()> on)
{
  float last = 0;
  float current = get_best()->fitness();
  bool on_opt = true;

  reset();

  do
  {
    last = current;
    optimize(count, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
    current = get_best()->fitness();

  } while(fabs(last - current) > dev && on_opt);
}

void algorithm::optimize_dev(float dev, std::function<bool()> on)
{
  reset();

  while(get_solutions()->fitness_dev() > dev)
  {
    optimize();
    on();
  }
}

void algorithm::optimize_eval(int count, std::function<bool()> on)
{
  reset();

  bool on_opt = true;
  long start = get_solutions()->get_evaluations();

  while(get_solutions()->get_evaluations() - start < count && on_opt)
  {
    optimize();
    on_opt = on();
  }
}

bool algorithm::is_maximization()
{
  return _maximization;
}

void algorithm::set_maximization(bool maximization)
{
  _maximization = maximization;
}

set<>* algorithm::get_solutions() const
{
  return _solutions;
}

long algorithm::get_iterations() const
{

}

void algorithm::set_params(int n, float* params)
{
  std::vector<float> vec_params(params, params + n);
  set_params(vec_params);
}

algorithm::~algorithm()
{
  delete _solutions;
}

} // namespace core
} // namespace dnn_opt
