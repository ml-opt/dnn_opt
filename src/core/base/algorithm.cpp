#include <cmath>
#include <core/base/algorithm.h>

namespace dnn_opt
{
namespace core
{

void algorithm::optimize(int count, std::function<void()> on)
{
  for(int i = 0; i < count; i++)
  {
    optimize();
    on();
  }
}

void algorithm::optimize_idev(int count, float dev, std::function<void()> on)
{
  float last = 0;
  float current = get_best()->fitness();

  do
  {
    last = current;
    optimize(count, on);
    current = get_best()->fitness();
  } while(fabs(last - current) > dev);
}

void algorithm::optimize_dev(float dev, std::function<void()> on)
{
  while(get_solutions()->fitness_dev() > dev)
  {
    optimize();
    on();
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

solution_set<>* algorithm::get_solutions() const
{
  return _solutions;
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
