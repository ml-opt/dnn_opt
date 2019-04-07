#include <math.h>
#include <cuda/solutions/griewangk.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

griewangk* griewangk::make(generator* generator, unsigned int size)
{
  auto* result = new griewangk(generator, size);

  result->init();

  return result;
}

float griewangk::calculate_fitness()
{
  float result = 0;

  return result;
}

griewangk::griewangk(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size)
{

}

griewangk::~griewangk()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
