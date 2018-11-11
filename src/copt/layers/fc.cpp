#include <cblas.h>
#include <algorithm>
#include <copt/layers/fc.h>

namespace dnn_opt
{
namespace copt
{
namespace layers
{

fc* fc::make(int in_dim, int out_dim, activation* activation)
{
  return new fc(in_dim, out_dim, activation);
}

void fc::ws(int size, const float* in, const float* params, float* out)
{
  /* Copy bias to out vector for each training pattern.
   *
   * Bias are located after regular parameters in @ref params vector.
   */
  for(int i = 0; i < size; i++)
  {
    int n = get_out_dim();
    int bias_pos = weight_size();
    int out_pos = i * get_out_dim();

    cblas_scopy(n, params + bias_pos, 1, out + out_pos, 1);
  }

  /* Perform C = 1.0 * A * B + 1.0 * C using SGEMM routine of BLAS. This
   * corresponds to the forward propagation of training patterns.
   *
   * C is the output vector of the layer.
   * A is a matrix that contains training patterns in a row by row fashion.
   * B is a matrix that contains layer parameters. Each neuron's parameters
   *   are arranged in a row by row fashion.
   *
   * m is the number of rows of A.
   * n is the number of columns of B and C (amount of neurons).
   * k is the number of columns of A and the number of rows of B.
   */

  int m = size;
  int n = get_out_dim();
  int k = get_in_dim();

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, in, k,
              params, k, 1.0, out, n);
}


fc *fc::clone()
{
  return fc::make(_in_dim, _out_dim, dynamic_cast<activation*>(_activation));
}

fc::fc(int in_dim, int out_dim, activation* activation)
: core::layers::fc(in_dim, out_dim, activation),
  core::layer(in_dim, out_dim, activation),
  layer(in_dim, out_dim, activation)
{

}

} // namespace layers
} // namespace copt
} // namespace dnn_opt
