#include <cmath>
#include <algorithm>
#include <core/activations/sigmoid.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

sigmoid* sigmoid::make()
{
  return new sigmoid();
}

void sigmoid::f(int size, const float* sum, float* out)
{
  for(int i = 0; i < size; i++)
  {
    out[i] = 1.0f / (1.0f + exp(-1.0f * sum[i]));
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
