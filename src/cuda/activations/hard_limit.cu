#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cuda/activations/hard_limit.h>

using namespace thrust;

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

namespace ops
{
namespace hard_limit
{

struct f
{
  f()
  {
    
  }

  __host__ __device__
  float operator()(const float& parameter) const
  {
    return parameter > 0 ? 1 : 0;
  }
};

}
}

hard_limit* hard_limit::make()
{
  return new hard_limit();
}

void hard_limit::f(int size, const float* sum, float* out)
{
  auto sum_ptr = device_pointer_cast(sum);
  auto out_ptr = device_pointer_cast(out);

  transform(sum_ptr, sum_ptr + size, out_ptr, ops::hard_limit::f());
}

}
}
}
