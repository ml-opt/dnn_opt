#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda/activations/relu.h>

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

namespace ops
{
namespace relu
{

struct f : public thrust::unary_function<float, float>
{
  __host__ __device__
  float operator()(const float& param) const
  {
    return fmaxf(0, param);
  }
};

}
}

relu* relu::make()
{
  return new relu();
}

void relu::f(int size, const float* sum, float* out)
{
  auto sum_ptr = thrust::device_pointer_cast(sum);
  auto out_ptr = thrust::device_pointer_cast(out);

  transform(sum_ptr, sum_ptr + size, out_ptr, ops::relu::f());
}

}
}
}
