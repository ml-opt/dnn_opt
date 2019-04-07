#include <cuda/base/layer.h>

namespace dnn_opt
{
namespace cuda
{

activation* layer::get_activation() const
{
  return _cuda_activation;
}

layer::layer(int in_dim, int out_dim, activation* activation)
: core::layer(in_dim, out_dim, activation)
{
  _cuda_activation = activation;
}

} // namespace cuda
} // namespace dnn_opt
