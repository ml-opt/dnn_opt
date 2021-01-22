#include <math.h>
#include <algorithm>
#include <core/solutions/bench/eggh.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

eggh* eggh::make(generator *generator, unsigned int size)
{
  auto* result = new eggh(generator, size);

  result->init();

  return result;
}

float eggh::calculate_fitness()
{
  float* params = get_params();
  float result = 0;
  int n = size();

  solution::calculate_fitness();

  for(int i = 0; i < n - 1; i++)
  {
    float br1 = params[i + 1] + 47.0f * sin(sqrt(abs(params[i + 1] + params[i] / 2.0f + 47.0f)));
    float br2 = params[i] * sin(sqrt(abs(params[i] - (params[i + 1] + 47.0f))));

    result += -(br1 + br2);
  }
  
  return result;
}

solution* eggh::clone()
{
  eggh* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

eggh::eggh(generator* generator, unsigned int size)
: solution(generator, size)
{

}

eggh::~eggh()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
