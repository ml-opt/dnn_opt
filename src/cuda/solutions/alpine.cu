#include <cmath>
#include <cuda/solutions/alpine.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

alpine* alpine::make(generator* generator, unsigned int size)
{
  auto* result = new alpine(generator, size);

  result->init();

  return result;
}

float alpine::calculate_fitness()
{
  float result = 0;

  return result;
}

alpine::alpine(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size)
{

}

alpine::~alpine()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
