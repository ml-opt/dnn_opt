#include <math.h>
#include <cuda/solutions/rosenbrock.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

rosenbrock* rosenbrock::make(generator* generator, unsigned int size)
{
  auto* result = new rosenbrock(generator, size);

  result->init();

  return result;
}

float rosenbrock::calculate_fitness()
{
  int n = size();
  float* params = get_params();
  float result = 0;

  solution::calculate_fitness();


  return result;
}

rosenbrock::rosenbrock(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size)
{

}

rosenbrock::~rosenbrock()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
