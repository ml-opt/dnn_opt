#include <stdexcept>
#include <core/algorithms/firefly.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void firefly::reset()
{
  _current_rand_influence = get_rand_influence();
}

void firefly::optimize()
{
  unsigned int n = get_solutions()->size();

  get_solutions()->sort(is_maximization());

  /** move the fireflies */
  for(int i = 0; i < n; i++)
  {
    auto* target = get_solutions()->get(i);

    for(int j = 0; j < n; j++)
    {
      auto* mover = get_solutions()->get(j);

      if(target->is_better_than(mover, is_maximization()))
      {
        move(j, i);
      }
    }
  }

  _current_rand_influence *= _rand_decay;
  _generator->set_min(-0.5 * _current_rand_influence);
  _generator->set_max(0.5 * _current_rand_influence);
}

solution* firefly::get_best()
{
  return get_solutions()->get_best(is_maximization());
}

void firefly::init()
{
  //delete created things

  _light_decay = 1;
  _rand_influence = 0.2;
  _current_rand_influence = 0.2;
  _rand_decay = 0.98;
  _init_bright = 1;

  float min = -0.5f * get_rand_influence();
  float max = 0.5f * get_rand_influence();

  _generator = generators::uniform::make(min, max);
  _r = new float[get_solutions()->get_dim()];
}

void firefly::move(int m, int t)
{
  int dim = get_solutions()->get_dim();
  float* mover = get_solutions()->get(m)->get_params();
  float* target = get_solutions()->get(t)->get_params();

  float dist = distance(m, t);
  float beta = get_init_bright() * exp(-1 * get_light_decay() * dist);

  _generator->generate(get_solutions()->get_dim(), _r);

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

  for (int i = 0; i < dim; i++)
  {
    sum += powf(target[i] - mover[i], 2.0f);
  }

  return sum;
}

firefly::~firefly()
{
  delete _generator;
  delete[] _r;
}

void firefly::set_params(std::vector<float> &params)
{
  if(params.size() != 3)
  {
    std::invalid_argument("algorithms::firefly set_params expect 3 values");
  }

  set_light_decay(params.at(0));
  set_init_bright(params.at(1));
  set_rand_influence(params.at(2));
}

float firefly::get_light_decay() const
{
  return _light_decay;
}

float firefly::get_init_bright() const
{
  return _init_bright;
}

float firefly::get_rand_influence() const
{
  return _rand_influence;
}

float firefly::get_rand_decay() const
{
  return _rand_decay;
}

void firefly::set_light_decay(float value)
{
  _light_decay = value;
}

void firefly::set_init_bright(float value)
{
  _init_bright = value;
}

void firefly::set_rand_influence(float value)
{
  _rand_influence = value;
  _current_rand_influence = value;
}

void firefly::set_rand_decay(float rand_decay)
{
  _rand_decay = rand_decay;
}

} // namespace algorithms
} // namepsace core
} // namespace dnn_opt
