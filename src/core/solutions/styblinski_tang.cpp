#include <math.h>
#include <core/solutions/styblinski_tang.h>

namespace dnn_opt
{
namespace core
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
  float result = 0;

  solution::calculate_fitness();

  for(int i = 0; i < size(); i++)
  {
    result += pow(get(i), 4) + 16 * pow(get(i), 2) + 5 * get(i);
  }

  return result / 2;
}

styblinski_tang::styblinski_tang(generator* generator, unsigned int size )
: solution(generator, size)
{

}

styblinski_tang::~styblinski_tang()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
