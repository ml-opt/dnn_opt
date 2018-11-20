#include <cmath>
#include <omp.h>
#include <copt/activations/elu.h>

namespace dnn_opt
{
namespace copt
{
namespace activations
{

elu* elu::make(float alpha)
{
  return new elu(alpha);
}

void elu::f(int size, const float* sum, float* out)
{
  #pragma omp simd
  for(int i = 0; i < size; i++)
  {
    if(sum[i] >= 0)
    {
      out[i] = sum[i];
    } else
    {
      out[i] = _alpha * (exp(out[i]) - 1);
    }
  }
}

elu::elu(float alpha)
: core::activations::elu(alpha)
{

}

} // namespace activations
} // namespace copt
} // namespace dnn_opt
