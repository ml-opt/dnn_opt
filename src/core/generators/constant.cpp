#include <stdexcept>
#include <core/generators/constant.h>

namespace dnn_opt
{
namespace core
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
  for(int i = 0; i < count; i++)
  {
    params[i] = _value;
  }
}

float constant::generate()
{
  return _value;
}


constant::constant(float value) : generator(value, value)
{
  _value = value;
}

constant::constant(float value, float min, float max) : generator(min, max)
{
  if(value < min || value > max)
  {
    throw std::out_of_range("value outside valid specified range");
  }

  _value = value;
}

constant::~constant()
{

}

} // namespace generators
} // namespace core
} // namespace dnn_opt
