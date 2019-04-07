#include <math.h>
#include <cuda/solutions/styblinski_tang.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

styblinski_tang* styblinski_tang::make(generator* generator, unsigned int size)
{
  auto* result = new styblinski_tang(generator, size);

  result->init();

  return result;
}

float styblinski_tang::calculate_fitness()
{
  int n = size();
  float* params = get_params();
  float result = 0;

  solution::calculate_fitness();

  return result / 2;
}

styblinski_tang::styblinski_tang(generator* generator, unsigned int size )
: solution(generator, size),
  core::solution(generator, size)
{

}

styblinski_tang::~styblinski_tang()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
