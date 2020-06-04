#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/csendes.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

csendes* csendes::make(generator *generator, unsigned int size)
{
  auto* result = new csendes(generator, size);

  result->init();

  return result;
}

float csendes::calculate_fitness()
{
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();
  
  int length = size();

  #pragma omp parallel for shared(length,params) reduction(+:result)
  for(int i = 0; i < length; i++)
  {
    result = pow(params[i], 6.0f) * (2.0f + sin(1.0f / params[i]));
  }

  return result;
}

csendes::csendes(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::csendes(generator, size)
{

}

csendes::~csendes()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
