#include <copt/generators/normal.h>

namespace dnn_opt
{
namespace copt
{
namespace generators
{

normal* normal::make(float mean, float dev)
{
  return new normal(mean, dev);
}

void normal::generate(int count, float* params)
{
  float ext = _max - _min;

  for(int i = 0; i < count; i++)
  {
    params[i] = (* _distribution)(*_generator) * ext + _min;
  }
}

float normal::generate()
{
  return (*_distribution)(*_generator);
}


normal::normal(float mean, float dev)
: generator(mean - dev, mean + dev),
  core::generator(mean - dev, mean + dev),
  core::generators::normal(mean, dev)
{

}

normal::~normal()
{

}


} // namespace generators
} // namespace copt
} // namespace dnn_opt
