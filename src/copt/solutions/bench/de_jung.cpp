#include <cblas.h>
#include <copt/solutions/bench/de_jung.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{
namespace bench
{

de_jung* de_jung::make(generator *generator, unsigned int size)
{
  auto* result = new de_jung(generator, size);

  result->init();

  return result;
}

float de_jung::calculate_fitness()
{  
  float result;

  #pragma omp parallel
  {
  result = cblas_sdot(size(), get_params(), 1, get_params(), 1);
  }

  solution::calculate_fitness();

  return result;
}

de_jung::de_jung(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::bench::de_jung(generator, size)
{

}

de_jung::~de_jung()
{

}

} // namespace bench
} // namespace solutions
} // namespace copt
} // namespace dnn_opt
