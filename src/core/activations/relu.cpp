#include <algorithm>
#include <core/activations/relu.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

relu* relu::make()
{
  return new relu();
}

void relu::f(int size, const float* sum, float* out)
{
  for(int i = 0; i < size; i++)
  {
    out[i] = std::max(0.0f, sum[i]);
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
