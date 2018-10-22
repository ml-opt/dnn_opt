#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cuda/activations/elu.h>

using namespace thrust;

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

namespace ops
{
namespace elu
{

struct f
{
  const float _alpha;

  f(float alpha) : _alpha(alpha)
  {

  }

  __host__ __device__
  float operator()(const float& param) const
  {
    if(param >= 0)
    {
      return param;
    } else
    {
      return _alpha * (expf(param) - 1);
    }
  }
};

}
}


elu* elu::make(float alpha)
{
  return new elu(alpha);
}

void elu::f(int size, const float* sum, float* out)
{
  auto sum_ptr = device_pointer_cast(sum);
  auto out_ptr = device_pointer_cast(out);

  transform(sum_ptr, sum_ptr + size, out_ptr, ops::elu::f(_alpha));
}

elu::elu(float alpha) : core::activations::elu(alpha)
{
  
}

}
}
}
