#include <cmath>
#include <copt/solutions/bench/alpine.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{


alpine* alpine::make(generator* generator, unsigned int size)
{
  auto* result = new alpine(generator, size);

  result->init();

  return result;
}

float alpine::calculate_fitness()
{
  int n = size();
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  #pragma omp simd
  for(int i = 0; i < n; i++)
  {
    result += std::fabs(params[i] * sin(params[i]) + 0.1 * params[i]);
  }

  return std::fabs(result);
}

alpine::alpine(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::alpine(generator, size)
{

}

alpine::~alpine()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
