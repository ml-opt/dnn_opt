#include <math.h>
#include <cuda/solutions/ackley.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

ackley* ackley::make(generator* generator, unsigned int size)
{
  auto* result = new ackley(generator, size);

  result->init();

  return result;
}

float ackley::calculate_fitness()
{
  return 0.0f;
}

ackley::ackley(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size)
{

}

ackley::~ackley()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
