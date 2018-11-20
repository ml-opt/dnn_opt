#include <cmath>
#include <core/solutions/step.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

step* step::make(generator* generator, unsigned int size)
{
  auto* result = new step(generator, size);

  result->init();

  return result;
}

float step::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += std::pow(std::floor(params[i]), 2);
  }

  return result;
}

step::step(generator* generator, unsigned int size)
: solution(generator, size)
{

}

step::~step()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
