#include <math.h>
#include <cuda/solutions/rastrigin.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

rastrigin* rastrigin::make(generator* generator, unsigned int size)
{
  auto* result = new rastrigin(generator, size);

  result->init();

  return result;
}

float rastrigin::calculate_fitness()
{
  int n = size();
  float* params = get_params();
  float result = 10 * size();

  return result;
}

rastrigin::rastrigin(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size)
{

}

rastrigin::~rastrigin()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
