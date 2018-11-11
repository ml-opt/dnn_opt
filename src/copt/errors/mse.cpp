#include <cassert>
#include <cmath>
#include <copt/errors/mse.h>

namespace dnn_opt
{
namespace copt
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

mse::mse()
{

}

} // namespace errors
} // namespace copt
} // namespace dnn_opt
