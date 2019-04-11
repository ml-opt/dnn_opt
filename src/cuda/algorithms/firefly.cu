#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda/algorithms/firefly.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

namespace ops
{
namespace firefly
{

struct dist_functor
{
public:

  dist_functor()
  {

  }

  template <typename tuple>
  __host__ __device__
  float operator()(tuple t) const
  {
    float source = thrust::get<0>(t);
    float target = thrust::get<1>(t);

    return powf(target - source, 2.0f);
  }
};

struct move_functor
{
  const float _beta;

  move_functor(float beta) : _beta(beta)
  {

  }

  template <typename tuple>
  __host__ __device__
  void operator()(tuple t) const
  {
    float source = thrust::get<0>(t);
    float target = thrust::get<1>(t);
    float random = thrust::get<2>(t);

    source += fmaf(_beta, target - source, random);

    thrust::get<0>(t) = source;
  }
};

} // namespace firefly
} // namespace ops

void firefly::init()
{
  cudaFree(_r);

  float min = -0.5f * this->get_rand_influence();
  float max = 0.5f * this->get_rand_influence();

  _generator = generators::uniform::make(min, max);
  cudaMalloc(&_r, get_solutions()->get_dim() * sizeof(float));
}

void firefly::move(int s, int t)
{
  auto s_ptr = thrust::device_pointer_cast(get_solutions()->get(s)->get_params());
  auto t_ptr = thrust::device_pointer_cast(get_solutions()->get(t)->get_params());
  auto r_ptr = thrust::device_pointer_cast(_r);

  int dim = get_solutions()->get_dim();
  float dist = distance(s, t);
  float beta = get_init_bright() * exp(-1 * get_light_decay() * dist);

  _generator->generate(dim, _r);

  auto tuple_begin = thrust::make_tuple(s_ptr, t_ptr, r_ptr);
  auto tuple_end = thrust::make_tuple(s_ptr + dim, t_ptr + dim, r_ptr + dim);

  auto begin = thrust::make_zip_iterator(tuple_begin);
  auto end = thrust::make_zip_iterator(tuple_end);

  thrust::for_each(begin, end, ops::firefly::move_functor(beta));
  get_solutions()->get(s)->set_constrains();
}

float firefly::distance(int s, int t)
{
  float result = 0;
  int dim = get_solutions()->get_dim();

  auto s_ptr = thrust::device_pointer_cast(get_solutions()->get(s)->get_params());
  auto t_ptr = thrust::device_pointer_cast(get_solutions()->get(t)->get_params());

  auto tuple_begin = thrust::make_tuple(s_ptr, t_ptr);
  auto tuple_end = thrust::make_tuple(s_ptr + dim, t_ptr + dim);

  auto begin = thrust::make_zip_iterator(tuple_begin);
  auto end = thrust::make_zip_iterator(tuple_end);

  result = thrust::transform_reduce(begin, end, ops::firefly::dist_functor(), 0.0f, thrust::plus<float>());

  return result;
}

firefly::~firefly()
{
  cudaFree(_r);

  /* avoids double free of core::firefly destructor */
  _r = 0;
}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt

