#include <math.h>
#include <core/solutions/michalewicz.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

michalewicz* michalewicz::make(generator* generator, unsigned int size)
{
  auto* result = new michalewicz(generator, size);

  result->init();

  return result;
}

float michalewicz::calculate_fitness()
{
  float result = 0;

  solution::calculate_fitness();
  for(int i = 0; i < size(); i++)
  {
    result += sin(get(i)) * pow(sin(i * pow(get(i), 2) / 3.141592653f), 20);
  }

  return result;
}

michalewicz::michalewicz(generator* generator, unsigned int size )
: solution(generator, size)
{

}

michalewicz::~michalewicz()
{

}


} // namespace solutions
} // namespace core
} // namespace dnn_opt
