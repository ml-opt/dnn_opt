#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/ackley.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

ackley* ackley::make(generator* generator, unsigned int size)
{
  auto* result = new ackley(generator, size);

  result->init();

  return result;
}

float ackley::calculate_fitness()
{
  int n = size();
  float summatory_1 = 0;
  float summatory_2 = 0;
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  #pragma omp simd
  for(int i = 0; i < n; i++)
  {
    summatory_1 += pow(params[i], 2);
    summatory_2 += cos(2 * 3.141592653f * params[i]);
  }

  result = -20 * exp(-0.2 * sqrt(summatory_1 / n)) ;
  result += -exp(summatory_2 / n) + 20 + 2.718281828f;

  return result;
}

ackley::ackley(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::ackley(generator, size)
{

}

ackley::~ackley()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
