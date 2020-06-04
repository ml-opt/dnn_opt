#include <cmath>
#include <omp.h>
#include <copt/solutions/bench/step.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

step* step::make(generator* generator, unsigned int size)
{
  auto* result = new step(generator, size);

  result->init();

  return result;
}

float step::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();

  #pragma omp simd
  for(int i = 0; i < n; i++)
  {
    result += std::pow(std::floor(params[i]), 2);
  }

  return result;
}

step::step(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::step(generator, size)
{

}

step::~step()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
