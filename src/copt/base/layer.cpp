#include <copt/base/layer.h>

namespace dnn_opt
{
namespace copt
{

layer::layer(int in_dim, int out_dim, activation* activation)
: core::layer(in_dim, out_dim, activation)
{

}

} // namespace copt
} // namespace dnn_opt
