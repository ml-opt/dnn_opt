#include <math.h>
#include <core/solutions/chung_reynolds.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

chung_r* chung_r::make(generator *generator, unsigned int size)
{
  auto* result = new brown(generator, size);

  result->init();

  return result;
}

float chung_r::calculate_fitness()
{
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  for(int i = 0; i < size(); i++)
  {
      result += pow(params[i], 2);
  }

  return pow(result, 2);
}

chung_r::chung_r(generator* generator, unsigned int size)
: solution(generator, size)
{

}

chung_r::~chung_r()
{

}

} // namespace solutions
} // namespace core
} // namespace dnn_opt
