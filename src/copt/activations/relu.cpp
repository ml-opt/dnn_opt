#include <algorithm>
#include <copt/activations/relu.h>

namespace dnn_opt
{
namespace copt
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
} // namespace copt
} // namespace dnn_opt
