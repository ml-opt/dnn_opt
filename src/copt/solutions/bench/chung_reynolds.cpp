#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/chung_reynolds.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

chung_r* chung_r::make(generator *generator, unsigned int size)
{
  auto* result = new chung_r(generator, size);

  result->init();

  return result;
}

float chung_r::calculate_fitness()
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

  return pow(result, 2.0f);
}

chung_r::chung_r(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::chung_r(generator, size)
{

}

chung_r::~chung_r()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
