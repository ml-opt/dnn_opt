#include <math.h>
#include <algorithm>
#include <core/solutions/bench/brown_function.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
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
  
  int length = size();

  for(int i = 0; i < length - 1; i++)
  {
    float result1 = pow(params[i], 2.0f * pow(params[i + 1], 2.0f) + 2.0f);
    float result2 = pow(params[i + 1], 2.0f * pow(params[i], 2.0f) + 2.0f);
    result += result1 + result2;
  }

  return result;
}

solution* brown::clone()
{
  brown* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

brown::brown(generator* generator, unsigned int size)
: solution(generator, size)
{

}

brown::~brown()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
