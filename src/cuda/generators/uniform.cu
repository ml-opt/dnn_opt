#include <random>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cuda/generators/uniform.h>

namespace dnn_opt
{
namespace cuda
{
namespace generators
{

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

}

uniform* uniform::make(float min, float max)
{
  return new uniform(min, max);
}

void uniform::generate(int count, float* params)
{
  auto ptr = thrust::device_pointer_cast(params);

  curandGenerateUniform(_gen, params, count);
  thrust::transform(ptr, ptr + count, ptr, range(_min, _max));
}

float uniform::generate()
{
  return core::generators::uniform::generate();
}

uniform::~uniform()
{
  curandDestroyGenerator(_gen);
}

uniform::uniform(float min, float max)
: generator(min, max),
  core::generator(min, max),
  core::generators::uniform(min, max)
{
  std::random_device device;

  curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(_gen, device());
}

}
}
}
