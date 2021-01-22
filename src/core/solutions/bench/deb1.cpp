#include <math.h>
#include <algorithm>
#include <core/solutions/bench/deb1.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

deb1* deb1::make(generator *generator, unsigned int size)
{
  auto* result = new deb1(generator, size);

  result->init();

  return result;
}

float deb1::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += pow(sin(5.0f * 3.141592653f * params[i]), 6.0f);
  }
  
  result = -1.0f * result / n;
  
  return result;
}

solution* deb1::clone()
{
  deb1* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

deb1::deb1(generator* generator, unsigned int size)
: solution(generator, size)
{

}

deb1::~deb1()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
