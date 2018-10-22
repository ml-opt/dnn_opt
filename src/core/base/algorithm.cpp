#include <core/base/algorithm.h>

namespace dnn_opt
{
namespace core
{

void algorithm::optimize(int count)
{
  for(int i = 0; i < count; i++)
  {
    optimize();
  }
}

void algorithm::optimize_iter_thrshold(int count, float variance)
{
  float last = 0;
  float current = get_best()->fitness();

  do
  {
    last = current;
    optimize(count);
    current = get_best()->fitness();
  } while(fabs(last - current) > variance);
}

void algorithm::optimize_dev_threshold(float dev)
{
  while(get_solutions()->fitness_dev() > dev) {
    optimize();
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
