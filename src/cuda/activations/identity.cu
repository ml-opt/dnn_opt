#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cuda/activations/identity.h>

using namespace thrust;

namespace dnn_opt
{
namespace cuda
{
namespace activations
{

namespace ops
{
namespace identity
{

struct f
{
  f()
  {
    
  }

  __host__ __device__
  float operator()(const float& parameter) const
  {
    return parameter;
  }
};

}
}

identity* identity::make()
{
  return new identity();
}

void identity::f(int size, const float* sum, float* out)
{
  auto sum_ptr = device_pointer_cast(sum);
  auto out_ptr = device_pointer_cast(out);

  transform(sum_ptr, sum_ptr + size, out_ptr, ops::identity::f());
}

}
}
}
