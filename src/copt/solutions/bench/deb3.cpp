#include <math.h>
#include <omp.h>
#include <copt/solutions/bench/deb3.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

deb3* deb3::make(generator *generator, unsigned int size)
{
  auto* result = new deb3(generator, size);

  result->init();

  return result;
}

float deb3::calculate_fitness()
{
  float result = 0;
  float pos = 0;
  float* params = get_params();

  solution::calculate_fitness();
  
  int length = size();

  #pragma omp parallel for shared(length, params) reduction(+:pos)
  for(int i = 0; i < length; i++)
  {
    pos = sin(5.0f * 3.14f * (pow(params[i], 0.75f) - 0.05f));
  }

  #pragma omp master
  {
  pos = pow(pos, 6.0f);
  result = -(1.0f / length) * pos;
  }

  return result;
}

deb3::deb3(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::deb3(generator, size)
{

}

deb3::~deb3()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
