#include <cmath>
#include <copt/solutions/alpine.h>

namespace dnn_opt
{
namespace copt
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

  for(int i = 0; i < n; i++)
  {
    result += params[i] * sin(params[i]) + 0.1 * params[i];
  }

  return result;
}

alpine::alpine(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::alpine(generator, size)
{

}

alpine::~alpine()
{

}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
