#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/brown_function.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

brown* brown::make(generator *generator, unsigned int size)
{
  auto* result = new brown(generator, size);

  result->init();

  return result;
}

float brown::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  float result1;
  float result2;

  solution::calculate_fitness();
  
  int length = size();
  #pragma omp parallel for shared(length, params) reduction(+:result)
  for(int i = 0; i < length - 1; i+=2)
  {
    result1 = pow(params[i], 2.0f * pow(params[i + 1], 2.0f) + 2.0f);
    result2 = pow(params[i + 1], 2.0f * pow(params[i], 2.0f) + 2.0f);
    result = result1 + result2;
  }


  return result;
}

brown::brown(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::brown(generator, size)
{

}

brown::~brown()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
