#include <math.h>
#include <algorithm>
#include <core/solutions/bench/cosine_mixture.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

cosine_m* cosine_m::make(generator *generator, unsigned int size)
{
  auto* result = new cosine_m(generator, size);

  result->init();

  return result;
}

float cosine_m::calculate_fitness()
{
  float result = 0;
  float result1 = 0;
  float result2 = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result1 += cos(5.0f * 3.141592653f * params[i]);
    result2 += pow(params[i], 2.0f);
  }

  result = -0.1f * result1 + result2;

  return result;
}

solution* cosine_m::clone()
{
  cosine_m* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

cosine_m::cosine_m(generator* generator, unsigned int size)
: solution(generator, size)
{

}

cosine_m::~cosine_m()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
