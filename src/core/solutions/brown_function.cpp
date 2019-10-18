#include <math.h>
#include <core/solutions/brown_function.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

brown* brown::make(generator *generator, unsigned int size)
{
  auto* result = new brown(generator, size);

  result->init();

  return result;
}

float brown::calculate_fitness()
{
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  for(int i = 0; i < size() - 1; i++)
  {
    float result1 = pow(params[i], 2.0f * pow(params[i + 1], 2) + 2.0f);
    float result2 = pow(params[i + 1], 2.0f * pow(params[i]) + 2.0f);
    result += result1 + result2;
  }

  return result;
}

brown::brown(generator* generator, unsigned int size)
: solution(generator, size)
{

}

brown::~brown()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
