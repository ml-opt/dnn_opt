#include <cmath>
#include <core/solutions/alpine.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

alpine* alpine::make(generator* generator, unsigned int size)
{
  auto* result = new alpine(generator, size);

  result->init();

  return result;
}

float alpine::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += std::fabs(params[i] * sin(params[i]) + 0.1 * params[i]);
  }

  return result;
}

alpine::alpine(generator* generator, unsigned int size)
: solution(generator, size)
{

}

alpine::~alpine()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
