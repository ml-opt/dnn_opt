#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/dixonp.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{
    
dixonp* dixonp::make(generator *generator, unsigned int size)
{
  auto* result = new dixonp(generator, size);

  result->init();

  return result;
}

float dixonp::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  float sum = 0;

  solution::calculate_fitness();
  
  int length = size();
  float binom = pow(params[0] + 1.0f, 2.0f);

  #pragma omp parallel for shared(length, params) reduction(+:sum)
  for(int i = 1; i < length; i++)
  {
    sum = i * pow(2 * pow(params[i], 2.0f) - params[i - 1], 2.0f);
  }

  result = binom + sum;
  
  return result;
}

dixonp::dixonp(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::dixonp(generator, size)
{

}

dixonp::~dixonp()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
