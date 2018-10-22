#include <math.h>
#include <core/solutions/schwefel.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

schwefel* schwefel::make(generator* generator, unsigned int size)
{
  auto* result = new schwefel(generator, size);

  result->init();

  return result;
}

float schwefel::calculate_fitness()
{
  solution::calculate_fitness();

  float result = 0;

  for(int i = 0; i < size(); i++)
  {
    result += - get(i) * sin(sqrt(fabs(get(i))));
  }

  return result;
}

schwefel::schwefel(generator* generator, unsigned int size )
: solution(generator, size)
{

}

schwefel::~schwefel()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
