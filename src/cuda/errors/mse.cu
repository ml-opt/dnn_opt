#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/errors/mse.h>

using namespace thrust;

namespace dnn_opt
{
namespace cuda
{
namespace errors
{

namespace ops
{
namespace mse
{

/**
 * @brief A thrust functor to calculate the squared difference between two
 * sequences of numbers.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct sq : public thrust::binary_function<float, float, float>
{
  template <typename Tuple>
  __host__ __device__
  float operator()(Tuple t) const
  {
    return powf(thrust::get<0>(t) - thrust::get<1>(t), 2);
  }
};

}
}

mse* mse::make()
{
  return new mse();
}

void mse::ff(int size, int dim, const float* out, const float* exp)
{
  int count = size * dim;

  auto out_ptr = thrust::device_pointer_cast(out);
  auto exp_ptr = thrust::device_pointer_cast(exp);

  auto tuple = thrust::make_tuple(exp_ptr, out_ptr);
  auto zip = thrust::make_zip_iterator(tuple);

  _accumulation += thrust::transform_reduce(zip, zip + count, ops::mse::sq(), 0.0f, thrust::plus<float>()) / dim;

  _size += size;
}

}
}
}
