#include <cmath>
#include <omp.h>
#include <algorithm>
#include <copt/activations/sigmoid.h>

namespace dnn_opt
{
namespace copt
{
namespace activations
{

sigmoid* sigmoid::make()
{
  return new sigmoid();
}

void sigmoid::f(int size, const float* sum, float* out)
{
  #pragma omp simd
  for(int i = 0; i < size; i++)
  {
    out[i] = 1 / (1 + exp(- 1 * sum[i]));
  }
}

} // namespace activations
} // namespace copt
} // namespace dnn_opt
