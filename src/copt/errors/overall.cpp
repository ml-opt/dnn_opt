#include <cmath>
#include <copt/errors/overall.h>

namespace dnn_opt
{
namespace copt
{
namespace errors
{

overall* overall::make()
{
  return new overall();
}

void overall::ff(int size, int dim, const float* out, const float* exp)
{
  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      if(exp[p + j] != out[p + j])
      {
        _accumulation++;
        break;
      }
    }
  }
  _size += size;
}


float overall::f()
{
  float result = _accumulation / _size;

  _accumulation = 0;
  _size = 0;

  return result;
}

overall::overall()
{

}

} // namespace errors
} // namespace copt
} // namespace dnn_opt
