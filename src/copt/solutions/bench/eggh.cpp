#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/eggh.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

eggh* eggh::make(generator *generator, unsigned int size)
{
  auto* result = new eggh(generator, size);

  result->init();

  return result;
}

float eggh::calculate_fitness()
{
  float br1 = 0;
  float br2 = 0;
  float* params = get_params();
  float result = 0;

  solution::calculate_fitness();
  
  int length = size();

  #pragma omp parallel for shared(length, params) reduction(+:result)
  for(int i = 0; i < length - 1; i+=2)
  {
    br1 = -(params[i + 1] + 47.0f) * sin(sqrt(abs(params[i + 1] + params[i]
            / 2.0f + 47.0f)));
    br2 = params[i] * sin(sqrt(abs(params[i] - (params[i + 1] + 47.0f))));
    result = br1 - br2;
  }
  
  return result;
}

eggh::eggh(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::eggh(generator, size)
{

}

eggh::~eggh()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
