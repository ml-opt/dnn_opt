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
  get_solutions()->generate();

  _light_decay = 1.0f;
  _rand_influence = 0.2f;
  _rand_decay = 0.98f;
  _init_bright = 1.0f;
}

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

      if(target->is_better_than(source, is_maximization()))
      {
        move(i, j);
      }
    }
  }

  _rand_influence *= _rand_decay;
  _generator->set_min(-0.5 * get_rand_influence());
  _generator->set_max(0.5 * get_rand_influence());
}

solution* firefly::get_best()
{
  return get_solutions()->get_best(is_maximization());
}

void firefly::init()
{
  //delete created things

  float min = -0.5f * get_rand_influence();
  float max = 0.5f * get_rand_influence();

  _generator = generators::uniform::make(min, max);
  _r = new float[get_solutions()->get_dim()];
}

void firefly::move(int s, int t)
{
  int dim = get_solutions()->get_dim();

  float* source = get_solutions()->get(s)->get_params();
  float* target = get_solutions()->get(t)->get_params();

  float dist = distance(s, t);
  float beta = get_init_bright() * exp(-1 * get_light_decay() * dist);

  _generator->generate(get_solutions()->get_dim(), _r);

  for(int i = 0; i < dim; i++)
  {
    source[i] +=  beta * (target[i] - source[i]) + _r[i];
  }

  get_solutions()->get(s)->set_constrains();
}

float firefly::distance(int s, int t)
{
  int dim = get_solutions()->get_dim();

  float result = 0;
  float* source = get_solutions()->get(s)->get_params();
  float* target = get_solutions()->get(t)->get_params();

  for (int i = 0; i < dim; i++)
  {
    result += powf(target[i] - source[i], 2.0f);
  }

  return result;
}

firefly::~firefly()
{
  delete _generator;
  delete[] _r;
}

void firefly::set_params(std::vector<float> &params)
{
  if(params.size() != 4)
  {
    std::invalid_argument("algorithms::firefly set_params expect 4 values");
  }

  set_light_decay(params.at(0));
  set_init_bright(params.at(1));
  set_rand_influence(params.at(2));
  set_rand_decay(params.at(3));
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
}

void firefly::set_rand_decay(float rand_decay)
{
  _rand_decay = rand_decay;
}

} // namespace algorithms
} // namepsace core
} // namespace dnn_opt
