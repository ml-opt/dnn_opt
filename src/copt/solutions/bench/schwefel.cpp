#include <cmath>
#include <omp.h>
#include <copt/solutions/bench/schwefel.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

schwefel* schwefel::make(generator* generator, unsigned int size)
{
  auto* result = new schwefel(generator, size);

  result->init();

  return result;
}

float schwefel::calculate_fitness()
{
  int n = size();
  float* params = get_params();

  solution::calculate_fitness();

  float result = 0;

  #pragma omp simd
  for(int i = 0; i < n; i++)
  {
    result += params[i] * sin(sqrt(fabs(params[i])));
  }

  return -1 * result / n;
}

schwefel::schwefel(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::schwefel(generator, size)
{

}

schwefel::~schwefel()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
