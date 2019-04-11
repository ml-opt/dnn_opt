#include <algorithm>
#include <core/layers/fc.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

fc* fc::make(int in_dim, int out_dim, activation* activation)
{
  return new fc(in_dim, out_dim, activation);
}

void fc::prop(int size, const float* in, const float* params, float* out)
{
  ws(size, in, params, out);
  get_activation()->f(size * get_out_dim(), out, out);
}

void fc::ws(int size, const float* in, const float* params, float* out)
{
  std::fill_n(out, size * get_out_dim(), 0.0f);

  /* transfer function */

  for(int i = 0; i < size; i++)
  {
    for(int k = 0; k < _in_dim; k++)
    {
      for(int j = 0; j < _out_dim; j++)
      {
        out[j * size + i] += in[k * size + i] * params[j * _in_dim + k];
      }
    }
  }

  /* including bias terms into transfer function */

  for(int i = 0; i < _out_dim; i++)
  {
    for(int j = 0; j < size; j++)
    {
      out[i * size + j] += params[_weight_size + i];
    }
  }
}

int fc::size() const
{
  return _size;
}

int fc::weight_size() const
{
  return _weight_size;
}

int fc::bias_size() const
{
  return get_out_dim();
}

layer* fc::clone()
{
  return fc::make(_in_dim, _out_dim, get_activation());
}

fc::fc(int in_dim, int out_dim, activation* activation)
: layer(in_dim, out_dim, activation)
{
  _weight_size  = in_dim * _out_dim;
  _size = _weight_size + out_dim;
}

} // namespace layers
} // namespace fc
} // namespace core
