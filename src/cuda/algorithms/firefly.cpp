#include <stdexcept>
#include <omp.h>
#include <copt/algorithms/firefly.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

void firefly::optimize()
{
  this->get_solutions()->sort(this->is_maximization());

  /** move the fireflies */
  for(int i = 0; i < this->get_solutions()->size(); i++)
  {
    for(int j = 0; j < this->get_solutions()->size(); j++)
    {
      auto* source = this->get_solutions()->get(i);
      auto* target = this->get_solutions()->get(j);

      if(target->is_better_than(source, this->is_maximization()))
      {
        move(i, j);
      }
    }
  }

  _current_rand_influence *= _rand_decay;
  _generator->set_min(-0.5 * _current_rand_influence);
  _generator->set_max(0.5 * _current_rand_influence);

  /*
  unsigned int n = get_solutions()->size();

  get_solutions()->sort(this->is_maximization());

  for(int i = 0; i < n; i++)
  {
    auto* target = this->get_solutions()->get(i);

    #pragma omp parallel for shared(n)
    for(int j = 0; j < n; j++)
    {
      auto* mover = get_solutions()->get(j);

      if(target->is_better_than(mover, is_maximization()))
      {
        move(j, i);
      }
    }
  }

  _rand_influence *= _rand_decay;
  _generator->set_min(-0.5 * get_rand_influence());
  _generator->set_max(0.5 * get_rand_influence());

  */
}

void firefly::move(int m, int t)
{
  int dim = get_solutions()->get_dim();
  float* mover = get_solutions()->get(m)->get_params();
  float* target = get_solutions()->get(t)->get_params();

  float dist = distance(m, t);
  float beta = get_init_bright() * exp(-1 * get_light_decay() * dist);

  _generator->generate(get_solutions()->get_dim(), _r);

  #pragma omp simd
  for(int i = 0; i < dim; i++)
  {
    mover[i] +=  beta * (target[i] - mover[i]) + _r[i];
  }

  get_solutions()->get(m)->set_constrains();
}

float firefly::distance(int m, int t)
{
  float sum = 0;
  int dim = get_solutions()->get_dim();
  float* mover = get_solutions()->get(m)->get_params();
  float* target = get_solutions()->get(t)->get_params();

  #pragma omp simd reduction (+:sum)
  for (int i = 0; i < dim; i++)
  {
    sum += powf(target[i] - mover[i], 2.0f);
  }

  return sum;
}

} // namespace algorithms
} // namepsace copt
} // namespace dnn_opt
