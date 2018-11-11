#include <copt/generators/uniform.h>

namespace dnn_opt
{
namespace copt
{
namespace generators
{

uniform* uniform::make(float min, float max)
{
  return new uniform(min, max);
}

void uniform::generate(int count, float* params)
{
  float ext = _max - _min;

  for(int i = 0; i < count; i++)
  {
    params[i] = (*_distribution)(*_generator) * ext + _min;
  }
}

float uniform::generate()
{
  return (*_distribution)(*_generator);
}

uniform::uniform(float min, float max)
: generator(min, max),
  core::generator(min, max),
  core::generators::uniform(min, max)
{

}

uniform::~uniform()
{

}

} // namespace generators
} // namespace copt
} // namespace dnn_opt
