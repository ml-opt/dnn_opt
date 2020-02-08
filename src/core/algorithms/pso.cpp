#include <stdexcept>
#include <core/algorithms/pso.h>
#include <iostream>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void pso::optimize()
{
  unsigned int n = get_solutions()->size();

  algorithm::optimize();

  for(int i = 0; i < n; i++)
  {
    update_speed(i);
    update_position(i);
    update_local(i);
  }

  if(_current_speed_param > get_min_speed_param())
  {
    _current_speed_param *= 0.99;
  }
}

solution* pso::get_best()
{
  return _best_so_far->get(_g_best);
}

void pso::update_speed(int idx)
{
  int dim = get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* best_so_far = _best_so_far->get(idx)->get_params();
  float* best = _best_so_far->get(_g_best)->get_params();
  float* speed = _speed->get(idx)->get_params();

  _generator->generate(2 * dim, _r);

  for(int i = 0; i < dim; i++)
  {
    speed[i] *= _current_speed_param;
    speed[i] += get_local_param() * _r[2 * i] * (best_so_far[i] - current[i]);
    speed[i] += get_global_param() * _r[2 * i + 1] * (best[i] - current[i]);
  }
}

void pso::update_position(int idx)
{
  int dim =  get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* speed = _speed->get(idx)->get_params();

  for(int i = 0; i < dim; i++)
  {
    current[i] += speed[i];
  }

  get_solutions()->get(idx)->set_constrains();
}

void pso::update_local(int idx)
{
  auto* current = get_solutions()->get(idx);
  auto* best_so_far = _best_so_far->get(idx);

  if(current->is_better_than(best_so_far, is_maximization()))
  {
    best_so_far->assign(current);
    update_global(idx);
  }
}

void pso::update_global(int idx)
{
  auto* current = _best_so_far->get(idx);
  auto* best = _best_so_far->get(_g_best);

  if(current->is_better_than(best, is_maximization()))
  {
    _g_best = idx;
  }
}

void pso::set_params(std::vector<float> &params)
{
  if(params.size() != 4)
  {
    std::invalid_argument("algorithms::pso set_params expect 4 values");
  }

  set_local_param(params.at(0));
  set_global_param(params.at(1));
  set_min_speed_param(params.at(2));
  set_max_speed_param(params.at(3));
}

float pso::get_local_param() const
{
  return _local_param;
}

float pso::get_global_param() const
{
  return _global_param;
}

float pso::get_min_speed_param() const
{
  return _min_speed_param;
}

float pso::get_max_speed_param() const
{
  return _max_speed_param;
}

void pso::set_local_param(float value)
{
  _local_param = value;
}

void pso::set_global_param(float value)
{
  _global_param = value;
}

void pso::set_max_speed_param(float value)
{
  _max_speed_param = value;
  _current_speed_param = value;
}

void pso::set_min_speed_param(float value)
{
  _min_speed_param = value;
}

void pso::reset()
{
  _current_speed_param = _max_speed_param;
  _speed->generate();

  get_solutions()->generate();

  for(int i = 0; i < get_solutions()->size(); i++)
  {
    _best_so_far->get(i)->assign(get_solutions()->get(i));
  }

  _g_best = _best_so_far->get_best_index(is_maximization());
}

void pso::init()
{
  if(_best_so_far != 0)
  {
    delete _best_so_far->clean();
  }
  if(_speed != 0)
  {
    delete _speed->clean();
  }
  delete _generator;
  delete _speed_generator;
  delete[] _r;

  float max_speed = 0.1f * get_solutions()->get(0)->get_generator()->get_ext();
  float min_speed = -1.0f * max_speed;

  _global_param = 0.8f;
  _local_param = 0.6f;
  _max_speed_param = 0.8;
  _min_speed_param = 0.1;
  _current_speed_param = _max_speed_param;

  _best_so_far = get_solutions()->clone();
  _g_best = _best_so_far->get_best_index(is_maximization());

  _generator = generators::uniform::make(0.0f, 1.0f);
  _speed_generator = generators::constant::make(0.0f, min_speed, max_speed);

  _r = new float[2 * get_solutions()->get_dim()];
  _speed = set<>::make(get_solutions()->size());

  for(int i = 0; i < get_solutions()->size(); i++)
  {
    _speed->add(solution::make(_speed_generator, get_solutions()->get_dim()));
  }

  _speed->generate();
}

pso::~pso()
{
  delete _generator;
  delete _speed_generator;
  delete _best_so_far->clean();
  delete _speed->clean();

  delete[] _r;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
