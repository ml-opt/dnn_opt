#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda/activations/tan_h.h>

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

namespace ops
{
namespace tan_h
{

struct f : public thrust::unary_function<float, float>
{
  __host__ __device__
  float operator()(const float& param) const
  {
    return tanhf(param);
  }
};

}
}

tan_h* tan_h::make()
{
  return new tan_h();
}

void tan_h::f(int size, const float* sum, float* out)
{
  auto sum_ptr = thrust::device_pointer_cast(sum);
  auto out_ptr = thrust::device_pointer_cast(out);

  thrust::transform(sum_ptr, sum_ptr + size, out_ptr, ops::tan_h::f());
}

}
}
}
