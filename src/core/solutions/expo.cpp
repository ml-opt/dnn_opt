#include <math.h>
#include <core/solutions/expo.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

expo* expo::make(generator *generator, unsigned int size)
{
  auto* result = new expo(generator, size);

  result->init();

  return result;
}

float expo::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  
  solution::calculate_fitness();
  
  int length = size();

  for(int i = 0; i < length; i++)
  {
    result += pow(params[i], 2.0f);
  }
  
  
  return -exp(-0.5f * result);
}

expo::expo(generator* generator, unsigned int size)
: solution(generator, size)
{

}

expo::~expo()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
