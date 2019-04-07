#include <stdexcept>
#include <omp.h>
#include <copt/algorithms/pso.h>

namespace dnn_opt
{
namespace copt
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

  #pragma omp parallel for simd
  for(int i = 0; i < dim; i++)
  {
    speed[i] *= current_param;
    speed[i] += local_param * _r[2 * i] * (best_so_far[i] - current[i]);
    speed[i] += global_param * _r[2 * i + 1] * (best[i] - current[i]);
  }

  _speed->get(idx)->set_constrains();
}

void pso::update_position(int idx)
{
  const unsigned int dim =  get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* speed = _speed->get(idx)->get_params();

  #pragma omp parallel for simd
  for(int i = 0; i < dim; i++)
  {
    current[i] += speed[i];
  }

  get_solutions()->get(idx)->set_constrains();
}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt
