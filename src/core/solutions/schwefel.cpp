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
  int n = size();
  float* params = get_params();

  solution::calculate_fitness();

  float result = 0;

  for(int i = 0; i < n; i++)
  {
    result += params[i] * sin(sqrt(fabs(params[i])));
  }

  return -1 * result / n;
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
