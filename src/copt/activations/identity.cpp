#include <copt/activations/identity.h>

namespace dnn_opt
{
namespace copt
{
namespace activations
{

identity* identity::make()
{
  return new identity();
}

void identity::f(int size, const float* sum, float* out)
{
  for(int i = 0; i < size; i++)
  {
    out[i] = sum[i];
  }
}

} // namespace activations
} // namespace copt
} // namespace dnn_opt
