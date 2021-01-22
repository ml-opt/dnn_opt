#include <math.h>
#include <algorithm>
#include <core/solutions/bench/expo.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

expo* expo::make(generator *generator, unsigned int size)
{
  auto* result = new expo(generator, size);

  result->init();

  return result;
}

float expo::calculate_fitness()
{
  float result = 0;
  float* params = get_params();
  int n = size();
  
  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += pow(params[i], 2.0f);
  }
  
  result = -exp(-0.5f * result);
  
  return result;
}

solution* expo::clone()
{
  expo* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

expo::expo(generator* generator, unsigned int size)
: solution(generator, size)
{

}

expo::~expo()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
