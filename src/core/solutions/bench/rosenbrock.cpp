#include <math.h>
#include <algorithm>
#include <core/solutions/bench/rosenbrock.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

rosenbrock* rosenbrock::make(generator* generator, unsigned int size)
{
  auto* result = new rosenbrock(generator, size);

  result->init();

  return result;
}

float rosenbrock::calculate_fitness()
{
  int n = size();
  float* params = get_params();

  solution::calculate_fitness();

  float result = 0;

  for(int i = 0; i < n - 1; i++)
  {
    result += 100.0f * pow(params[i + 1] - pow(params[i], 2.0f), 2.0f) + pow(params[i] - 1.0f, 2.0f);
  }

  return result;
}

solution* rosenbrock::clone()
{
  rosenbrock* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

rosenbrock::rosenbrock(generator* generator, unsigned int size )
: solution(generator, size)
{

}

rosenbrock::~rosenbrock()
{

}

} // namespace brench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
