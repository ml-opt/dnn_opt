#include <math.h>
#include <algorithm>
#include <core/solutions/bench/deb3.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

deb3* deb3::make(generator *generator, unsigned int size)
{
  auto* result = new deb3(generator, size);

  result->init();

  return result;
}

float deb3::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();

  solution::calculate_fitness();
  
  for(int i = 0; i < n; i++)
  {
    result += pow(sin(5.0f * 3.141592653f * (pow(params[i], 0.75f) - 0.05f)), 6.0f);
  }

  result = -1.0f * result / n;
  
  return result;
}

solution* deb3::clone()
{
  deb3* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

deb3::deb3(generator* generator, unsigned int size)
: solution(generator, size)
{

}

deb3::~deb3()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
