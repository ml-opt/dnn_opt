#include <cmath>
#include <omp.h>
#include <copt/solutions/bench/griewangk.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

griewangk* griewangk::make(generator* generator, unsigned int size)
{
  auto* result = new griewangk(generator, size);

  result->init();

  return result;
}

float griewangk::calculate_fitness()
{
  int n = size();
  float* params = get_params();
  float summatory = 0;
  float multiplier = 1;
  float result = 0;

  solution::calculate_fitness();

  #pragma omp simd
  for(int i = 0; i < n; i++)
  {
    summatory  += pow(params[i], 2.0) / 4000;
    multiplier *= cos(params[i] / sqrt(i + 1));
  }

  result = summatory - multiplier + 1;

  return result;
}

griewangk::griewangk(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::griewangk(generator, size)
{

}

griewangk::~griewangk()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
