#include <math.h>
#include <core/solutions/eggh.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

eggh* eggh::make(generator *generator, unsigned int size)
{
  auto* result = new eggh(generator, size);

  result->init();

  return result;
}

float eggh::calculate_fitness()
{
  float br1 = 0;
  float br2 = 0;
  float* params = get_params();
  float result = 0;

  solution::calculate_fitness();
  
  int length = size();

  for(int i = 0; i < length - 1.0f; i++)
  {
    br1 = -(params[i + 1] + 47.0f) * sin(sqrt(abs(params[i + 1] + params[i]/2.0f
          + 47.0f)));
    br2 = params[i] * sin(sqrt(abs(params[i] - (params[i + 1] + 47.0f))));
    result += br1 - br2; 
  }
  
  return result;
}

eggh::eggh(generator* generator, unsigned int size)
: solution(generator, size)
{

}

eggh::~eggh()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
