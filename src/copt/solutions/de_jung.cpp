#include <cblas.h>
#include <copt/solutions/de_jung.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{

de_jung* de_jung::make(generator *generator, unsigned int size)
{
  auto* result = new de_jung(generator, size);

  result->init();

  return result;
}

float de_jung::calculate_fitness()
{
  float result = cblas_sdot(size(), get_params(), 1, get_params(), 1);

  return result;
}

de_jung::de_jung(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::de_jung(generator, size)
{

}

de_jung::~de_jung()
{

}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
