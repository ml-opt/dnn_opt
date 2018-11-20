#include <stdexcept>
#include <omp.h>
#include <copt/generators/constant.h>

namespace dnn_opt
{
namespace copt
{
namespace generators
{

constant* constant::make(float value)
{
  return new constant(value);
}

constant* constant::make(float value, float min, float max)
{
  return new constant(value, min, max);
}

void constant::generate(int count, float* params)
{
  #pragma omp simd
  for(int i = 0; i < count; i++)
  {
    params[i] = _value;
  }
}

float constant::generate()
{
  return _value;
}

constant::constant(float value)
: generator(value, value),
  core::generator(value, value),
  core::generators::constant(value)
{
  _value = value;
}

constant::constant(float value, float min, float max)
: generator(min, max),
  core::generator(min, max),
  core::generators::constant(value, min, max)
{

}

constant::~constant()
{

}

} // namespace generators
} // namespace copt
} // namespace dnn_opt
