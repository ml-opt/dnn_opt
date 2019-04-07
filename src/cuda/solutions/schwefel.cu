#include <math.h>
#include <cuda/solutions/schwefel.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

schwefel* schwefel::make(generator* generator, unsigned int size)
{
  auto* result = new schwefel(generator, size);

  result->init();

  return result;
}

float schwefel::calculate_fitness()
{
  int n = size();
  float* params = get_params();

  solution::calculate_fitness();

  float result = 0;

  return -1 * result / n;
}

schwefel::schwefel(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size)
{

}

schwefel::~schwefel()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
