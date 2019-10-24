#include <math.h>
#include <core/solutions/csendes.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

csendes* csendes::make(generator *generator, unsigned int size)
{
  auto* result = new csendes(generator, size);

  result->init();

  return result;
}

float csendes::calculate_fitness()
{
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();
  
  int length = size();

  for(int i = 0; i < length; i++)
  {
    result += pow(params[i], 6.0f) * (2.0f + sin(1.0f / params[i]));
  }
  
  return result;
}

csendes::csendes(generator* generator, unsigned int size)
: solution(generator, size)
{

}

csendes::~csendes()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
