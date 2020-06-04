#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/expo.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

expo* expo::make(generator *generator, unsigned int size)
{
  auto* result = new expo(generator, size);

  result->init();

  return result;
}

float expo::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  
  solution::calculate_fitness();
  
  int length = size();

  #pragma omp parallel for shared(length, params) reduction(+:result)
  for(int i = 0; i < length; i++)
  {
    result = pow(params[i], 2.0f);
  }
  
  return -exp(-0.5f * result);
}

expo::expo(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::expo(generator, size)
{

}

expo::~expo()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
