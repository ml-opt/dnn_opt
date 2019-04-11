#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <vector>
#include <stdexcept>
#include <cuda/algorithms/cuckoo.h>

#include <iostream>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

namespace
{

struct gen_functor
{
public:

  gen_functor(float scale, float levy) : _scale(scale), _levy(levy)
  {

  }

  template <typename tuple>
  __host__ __device__
  float operator()(tuple t) const
  {
    float best = thrust::get<0>(t);
    float cuckoo = thrust::get<1>(t);
    float params = thrust::get<2>(t);
    float r = thrust::get<3>(t);

    return params + _scale * _levy * (best - cuckoo) * r;
  }

private:

  float _scale;
  float _levy;
};

}

void cuckoo::generate_new_cuckoo(int cuckoo_idx)
{
  int dim = get_solutions()->get_dim();

  auto cuckoo_ptr = thrust::device_pointer_cast(get_solutions()->get(cuckoo_idx)->get_params());
  auto best_ptr = thrust::device_pointer_cast(get_solutions()->get_best(is_maximization())->get_params());
  auto params_ptr = thrust::device_pointer_cast(_updated->get_params());
  auto r_ptr = thrust::device_pointer_cast(_r);

  float v = _nd_1->generate();
  float u = _nd_o->generate();
  float levy = u / powf(fabs(v), 1 / _levy);

  _nd_1->generate(dim, _r);

  auto tuple_begin = thrust::make_tuple(cuckoo_ptr, best_ptr, params_ptr, r_ptr);
  auto tuple_end = thrust::make_tuple(cuckoo_ptr + dim, best_ptr + dim, params_ptr + dim, r_ptr + dim);

  auto begin = thrust::make_zip_iterator(tuple_begin);
  auto end = thrust::make_zip_iterator(tuple_end);

  thrust::for_each(begin, end, gen_functor(_scale, levy));

  _updated->set_modified(true);
}

void cuckoo::init()
{
  cudaFree(_r);

  _scale = 0.8;
  _levy = 0.8;
  _replacement = 0.3;

  /** mantegna algorithm to calculate levy steep size */

  float dividend = tgamma(1 + _levy) * sin(3.14159265f * _levy / 2);
  float divisor = tgamma((1 + _levy) / 2) * _levy * pow(2, (_levy - 1) / 2);
  float omega = pow(dividend / divisor , 1 / _levy);

  _nd_1 = generators::normal::make(0, 1);
  _nd_o = generators::normal::make(0, omega);
  _selector = generators::uniform::make(0, get_solutions()->size());
  _updated = get_solutions()->get(0)->clone();

  cudaMalloc(&_r, get_solutions()->get_dim() * sizeof(float));
}

cuckoo::~cuckoo()
{
  cudaFree(_r);

  /* avoids double free of core::cuckoo destructor */
  _r = 0;
}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt
