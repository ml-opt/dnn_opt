#include <cassert>
#include <math.h>
#include <core/errors//mse.h>

namespace dnn_opt
{
namespace core
{
namespace errors
{

mse* mse::make()
{
  return new mse();
}

void mse::ff(int size, int dim, const float* out, const float* exp)
{
  for(int i = 0; i < size; i++)
  {
    float squared_sum = 0;
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      squared_sum += pow(exp[p + j] - out[p + j], 2);
    }

    _accumulation += squared_sum / dim;
  }
  _size += size;
}

float mse::f()
{
  assert(_size != 0);

  float result = _accumulation / _size;

  _accumulation = 0;
  _size = 0;

  return result;
}

mse::mse()
{
  _accumulation = 0;
  _size = 0;
}

} // namespace errors
} // namespace core
} // namespace dnn_opt
