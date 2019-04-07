#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda/algorithms/pso.h>
#include <iostream>
namespace
{

struct speed_updater
{
public:

  speed_updater(float current, float local, float global)
  : _current(current),
    _local(local),
    _global(global)
  {

  }

  template <typename Tuple>
  __host__ __device__
  float operator()(Tuple t) const
  {
    float speed = thrust::get<0>(t);
    float current = thrust::get<1>(t);
    float best = thrust::get<2>(t);
    float best_so_far = thrust::get<3>(t);
    float r1 = thrust::get<4>(t);
    float r2 = thrust::get<5>(t);

    float result = _current * speed +
                   _local * r1 * (best_so_far - current) +
                   _global * r2 * (best - current);

    return result;
  }

private:

  const float _current;
  const float _local;
  const float _global;
};

} // namespace

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

void pso::update_speed(int idx)
{
  const unsigned int dim =  get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* best_so_far = _best_so_far->get(idx)->get_params();
  float* best = _best_so_far->get(_g_best)->get_params();
  float* speed = _speed->get(idx)->get_params();

  const float current_param = _current_speed_param;
  const float local_param = get_local_param();
  const float global_param = get_global_param();

  _generator->generate(2 * dim, _r);  

  auto current_ptr = thrust::device_pointer_cast(current);
  auto best_so_far_ptr = thrust::device_pointer_cast(best_so_far);
  auto best_ptr = thrust::device_pointer_cast(best);
  auto speed_ptr = thrust::device_pointer_cast(speed);
  auto r1_ptr = thrust::device_pointer_cast(_r);
  auto r2_ptr = thrust::device_pointer_cast(_r + dim);

  auto op = speed_updater(current_param, local_param, global_param);

  auto tuple = thrust::make_tuple(speed_ptr, current_ptr, best_ptr,
                                  best_so_far_ptr, r1_ptr, r2_ptr);
  auto ziped = thrust::make_zip_iterator(tuple);

  thrust::transform(ziped, ziped + dim, speed_ptr, op);

  _speed->get(idx)->set_constrains();
}

void pso::update_position(int idx)
{
  const unsigned int dim =  get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* speed = _speed->get(idx)->get_params();

  auto current_ptr = thrust::device_pointer_cast(current);
  auto speed_ptr = thrust::device_pointer_cast(speed);

  thrust::transform(current_ptr, current_ptr + dim, speed_ptr, current_ptr,
                    thrust::plus<float>());

  get_solutions()->get(idx)->set_constrains();
}

void pso::init()
{
  // TODO: delete created things...

  float max_speed = 0.1f * get_solutions()->get(0)->get_generator()->get_ext();
  float min_speed = -1.0f * max_speed;

  _global_param = 0.8f;
  _local_param = 0.6f;
  _max_speed_param = 0.8;
  _min_speed_param= 0.1;
  _current_speed_param = _max_speed_param;

  _best_so_far = get_solutions()->clone();
  _g_best = _best_so_far->get_best_index(is_maximization());

  _generator = generators::uniform::make(0.0f, 1.0f);
  auto cuda_speed_generator = generators::constant::make(0.0f, min_speed, max_speed);
  _speed_generator = cuda_speed_generator;

  cudaMalloc(&_r, 2 * get_solutions()->get_dim() * sizeof(float));

  _speed = core::set<>::make(get_solutions()->size());

  for(int i = 0; i < get_solutions()->size(); i++)
  {
    _speed->add(solution::make(cuda_speed_generator, get_solutions()->get_dim()));
  }

  _speed->generate();
}

pso::~pso()
{
  cudaFree(_r);

  _r = 0;
}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt
