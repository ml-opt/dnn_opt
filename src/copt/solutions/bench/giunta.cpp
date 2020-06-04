#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/giunta.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

giunta* giunta::make(generator *generator, unsigned int size)
{
  auto* result = new giunta(generator, size);

  result->init();

  return result;
}

float giunta::calculate_fitness()
{
  float result = 0;
  float term1 = 0;
  float term2 = 0;
  float term3 = 0;
  float* params = get_params();
  
  solution::calculate_fitness();
  
  int length = 2;

  #pragma omp parallel for shared(length, params) reduction(+:result)
  for(int i = 0; i < length; i++)
  {
  term1 = sin((16.0f / 15.0f) * params[i] - 1.0f);
  term2 = pow(sin((16.0f / 15.0f) * params[i] - 1.0f), 2.0f);
  term3 = 1.0f / 50.0f * sin(4.0f * ((16.0f / 15.0f) * params[i] - 1.0f));
  result = term1 + term2 + term3;
  }

  return 0.6f + result;
}

giunta::giunta(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::giunta(generator, size)
{

}

giunta::~giunta()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
