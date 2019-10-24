#include <math.h>
#include <core/solutions/cosine_mixture.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

cosine_m* cosine_m::make(generator *generator, unsigned int size)
{
  auto* result = new cosine_m(generator, size);

  result->init();

  return result;
}

float cosine_m::calculate_fitness()
{
  float result1 = 0;
  float result2 = 0;
  float* params = get_params();

  solution::calculate_fitness();
  
  int length = size();

  for(int i = 0; i < length; i++)
  {
    result1 += cos(5.0f * 3.14f * params[i]);
  }
  
  result1 *= -(0.1f);
  
  for(int j = 0; j < size(); j++)
  {
    result2 += pow(params[j], 2.0f);
  }
  
  return result1 - result2;
}

cosine_m::cosine_m(generator* generator, unsigned int size)
: solution(generator, size)
{

}

cosine_m::~cosine_m()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
