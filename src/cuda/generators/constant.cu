#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda/generators/constant.h>

namespace dnn_opt
{
namespace cuda
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
  auto params_ptr = thrust::device_pointer_cast(params);
  thrust::fill_n(params_ptr, count, _value);
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
} // namespace cuda
} // namespace dnn_opt
