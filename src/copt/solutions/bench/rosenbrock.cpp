#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/rosenbrock.h>

namespace dnn_opt
{
namespace copt
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
  float result = 0;

  solution::calculate_fitness();

  #pragma omp simd
  for(int i = 0; i < n - 1; i++)
  {
    result += 100 * pow(params[i + 1] - pow(params[i], 2), 2) + pow(params[i] - 1, 2);
  }

  return result;
}

rosenbrock::rosenbrock(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::rosenbrock(generator, size)
{

}

rosenbrock::~rosenbrock()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
