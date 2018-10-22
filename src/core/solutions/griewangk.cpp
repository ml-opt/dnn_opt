#include <math.h>
#include <core/solutions/griewangk.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

griewangk* griewangk::make(generator* generator, unsigned int size)
{
  auto* result = new griewangk(generator, size);

  result->init();

  return result;
}

float griewangk::calculate_fitness()
{
  float summatory     = 0;
  float multiplier    = 1;
  float result        = 0;

  solution::calculate_fitness();
  for(int i = 0; i < size(); i++)
  {
    summatory  += get(i);
    multiplier *= cos(get(i) / sqrt(i));
  }
  result = summatory / 4000 - multiplier + 1;

  return result;
}

griewangk::griewangk(generator* generator, unsigned int size)
: solution(generator, size)
{

}

griewangk::~griewangk()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
