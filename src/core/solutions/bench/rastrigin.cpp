#include <math.h>
#include <core/solutions/bench/rastrigin.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

rastrigin* rastrigin::make(generator* generator, unsigned int size)
{
  auto* result = new rastrigin(generator, size);

  result->init();

  return result;
}

float rastrigin::calculate_fitness()
{
  int n = size();
  float* params = get_params();
  float result = 10 * size();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += pow(params[i], 2) - 10 * cos(2 * 3.141592653f * params[i]);
  }

  return result;
}

rastrigin::rastrigin(generator* generator, unsigned int size )
: solution(generator, size)
{

}

rastrigin::~rastrigin()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
