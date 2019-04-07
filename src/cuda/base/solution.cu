#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <cuda/base/solution.h>

using namespace thrust;

namespace dnn_opt
{
namespace cuda
{

namespace ops
{
namespace solution
{

struct constrains : public thrust::unary_function<float, float>
{
  const float _min;
  const float _max;

  constrains(float min, float max) : _min(min), _max(max)
  {

  }

  __host__ __device__
  float operator()(const float& param) const
  {
    return fminf(_max, fmaxf(_min, param));
  }
};

} // namespace solution
} // namespace ops

solution* solution::make(generator* generator, unsigned int size)
{
  auto* result = new solution(generator, size);

  result->init();

  return result;
}

void solution::assign(core::solution* s)
{
  if(assignable(s) == false)
  {
    throw new std::invalid_argument("solution is not compatible");
  }

  auto s_params_ptr = thrust::device_pointer_cast(s->get_params());
  auto params_ptr = thrust::device_pointer_cast(get_params());

  thrust::copy_n(s_params_ptr, size(), params_ptr);

  _fitness = s->fitness();
  _evaluations = s->get_evaluations();

  set_modified(false);
}

generator* solution::get_generator() const
{
  return _dev_generator;
}

void solution::set_constrains()
{
  float min = get_generator()->get_min();
  float max = get_generator()->get_max();
  auto ptr = thrust::device_pointer_cast(get_params());

  thrust::transform(ptr, ptr + size(), ptr, ops::solution::constrains(min, max));

  set_modified(true);
}

void solution::init()
{
  cudaFree(_params);
  cudaMalloc(&_params, size() * sizeof(float));

  _evaluations = 0;
  _modified = true;
}

solution::solution(generator* generator, unsigned int size)
: core::solution(generator, size)
{
  _dev_generator = generator;
}

solution::~solution()
{
  cudaFree(_params);

  /* avoid double free from core::solution destructor */
  _params = 0;
}

} // namespace cuda
} // namespace dnn_opt
