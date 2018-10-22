#include <stdexcept>
#include <core/base/generator.h>

namespace dnn_opt
{
namespace core
{

float generator::get_min()
{
  return _min;
}

float generator::get_max()
{
  return _max;
}

void generator::set_min(float min)
{
  _min = min;
}

void generator::set_max(float max)
{
  _max = max;
}

generator::generator(float min, float max)
{
  if(min > max)
  {
    throw new std::invalid_argument("invalid range specification");
  }

  _min = min;
  _max = max;
}

generator::~generator()
{

}

} // namespace core
} // namespace dnn_opt
