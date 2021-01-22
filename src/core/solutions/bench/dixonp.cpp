#include <math.h>
#include <algorithm>
#include <core/solutions/bench/dixonp.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{
    
dixonp* dixonp::make(generator *generator, unsigned int size)
{
  auto* result = new dixonp(generator, size);

  result->init();

  return result;
}

float dixonp::calculate_fitness()
{
  float* params = get_params();
  float result = pow(params[0] - 1.0f, 2.0f);
  int n = size();

  solution::calculate_fitness();

  for(int i = 1; i < n; i++)
  {
    /* Notice the correction in the initial multiplication (i + 1) for cero-based indexing */
    result += (i + 1) * pow(2.0f * pow(params[i], 2.0f) - params[i - 1], 2.0f);
  }
  
  return result;
}

solution* dixonp::clone()
{
  dixonp* clon = make(get_generator(), size());

  std::copy_n(get_params(), size(), clon->get_params());

  return clon;
}

dixonp::dixonp(generator* generator, unsigned int size)
: solution(generator, size)
{

}

dixonp::~dixonp()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
