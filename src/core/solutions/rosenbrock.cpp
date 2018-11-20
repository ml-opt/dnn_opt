#include <math.h>
#include <core/solutions/rosenbrock.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
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
  float result = 0;

  solution::calculate_fitness();

  for(int i = 0; i < n - 1; i++)
  {
    result += 100 * pow(params[i + 1] - pow(params[i], 2), 2) + pow(params[i] - 1, 2);
  }

  return result;
}

rosenbrock::rosenbrock(generator* generator, unsigned int size )
: solution(generator, size)
{

}

rosenbrock::~rosenbrock()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
