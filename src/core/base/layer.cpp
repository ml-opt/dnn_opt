#include <core/base/layer.h>

namespace dnn_opt
{
namespace core
{

int layer::get_in_dim() const
{
  return _in_dim;
}

int layer::get_out_dim() const
{
  return _out_dim;
}

activation* layer::get_activation() const
{
  return _activation;
}

layer::layer(int in_dim, int out_dim, activation* activation)
{
  _in_dim = in_dim;
  _out_dim = out_dim;
  _activation = activation;
}

layer::~layer()
{

}

} // namespace core
} // namespace dnn_opt
