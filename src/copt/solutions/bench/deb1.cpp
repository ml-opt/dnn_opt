#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/deb1.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

deb1* deb1::make(generator *generator, unsigned int size)
{
  auto* result = new deb1(generator, size);

  result->init();

  return result;
}

float deb1::calculate_fitness()
{
  float result = 0;
  float pos = 0;
  float* params = get_params();

  solution::calculate_fitness();
  
  int length = size();

  #pragma omp parallel for shared(length,params) reduction(+:pos)
  for(int i = 0; i < length; i++)
  {
    pos = sin(5.0f * 3.14f * params[i]);
  }

  #pragma omp master
  {
  pos = pow(pos, 6.0f);
  result = -(1.0f / length) * pos;
  }
  return result;
}

deb1::deb1(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::deb1(generator, size)
{

}

deb1::~deb1()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
