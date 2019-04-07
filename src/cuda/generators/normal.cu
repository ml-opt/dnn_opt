#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cuda/generators/normal.h>

namespace
{

struct range
{
  const float _min;
  const float _ext;

  range(float min, float max)
  : _min(min),
    _ext(max - min)
  {

  }

  __host__ __device__
  float operator()(const float& param) const
  {
    return param * _ext + _min;
  }
};

} // namespce


namespace dnn_opt
{
namespace cuda
{
namespace generators
{

normal* normal::make(float mean, float dev)
{
  return new normal(mean, dev);
}

void normal::generate(int count, float* params)
{
  auto ptr = thrust::device_pointer_cast(params);

  curandGenerateNormal(_gen, params, count, _mean, _dev);
  thrust::transform(ptr, ptr + count, ptr, range(_min, _max));
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
  std::random_device device;

  curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(_gen, device());
}

normal::~normal()
{
  curandDestroyGenerator(_gen);
}


} // namespace generators
} // namespace cuda
} // namespace dnn_opt
