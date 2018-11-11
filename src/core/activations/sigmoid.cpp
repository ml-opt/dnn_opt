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
    out[i] = 1 / (1 + exp(- 1 * sum[i]));
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
