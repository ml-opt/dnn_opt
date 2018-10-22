#include <cuda/base/layer.h>

namespace dnn_opt
{
namespace cuda
{

layer::layer(int in_dim, int out_dim, activation* activation)
: core::layer(in_dim, out_dim, activation)
{

}

} // namespace cuda
} // namespace dnn_opt
