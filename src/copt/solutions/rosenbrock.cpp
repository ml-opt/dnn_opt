#include <math.h>
#include <copt/solutions/rosenbrock.h>

namespace dnn_opt
{
namespace copt
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
  float result = solution::calculate_fitness();

  for(int i = 0; i < size() - 1; i++)
  {
    result += 100 * pow(get(i + 1) - pow(get(i), 2), 2) + pow(get(i) - 1, 2);
  }

  return result;
}

rosenbrock::rosenbrock(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::rosenbrock(generator, size)
{

}

rosenbrock::~rosenbrock()
{

}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
