#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
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

void uniform::set_constraints(int count, float* params)
{
  float min = get_min();
  float max = get_max();

  for(int i = 0; i < count; i++)
  {
    params[i] = std::max(min, std::min(max, params[i]));
  }
}

uniform::uniform(float min, float max) : generator(min, max)
{
  std::random_device device;

  _generator = new std::mt19937(device());
  _distribution = new std::uniform_real_distribution<>(0, 1);
}

uniform::~uniform()
{
  delete _generator;
  delete _distribution;
}

} // namespace generators
} // namespace core
} // namespace dnn_opt
