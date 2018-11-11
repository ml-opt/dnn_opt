#include <cmath>
#include <copt/activations/tan_h.h>

namespace dnn_opt
{
namespace copt
{
namespace activations
{

tan_h* tan_h::make()
{
  return new tan_h();
}

void tan_h::f(int size, const float* sum, float* out)
{
  for(int i = 0; i < size; i++)
  {
    out[i] = tanh(sum[i]);
  }
}

} // namespace activations
} // namespace copt
} // namespace dnn_opt
