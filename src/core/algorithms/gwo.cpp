#include <stdexcept>
#include <core/algorithms/gwo.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void gwo::optimize()
{
  for (int i = 0; i < get_solutions()->size(); i++)
  {
    _generator->generate(2*_dim,_r1);
    _generator->generate(2*_dim,_r2);
    auto* current = get_solutions()->get(i)->get_params();

    for (int j = 0; j < _dim; j++)
    {
      _A1[j] = 2 * _a * _r1 - _a;
      _C1[j] = 2 * _r2[j];
      _Da[j] = abs(_C1[j] * _alpha->get_params()[j] - current[i]);
      _X1[j] = _alpha->get_params()[j] - _A1[j] * _Da[j];
      _A1[j] = 2 * _a * _r1 - _a;
      _C1[j] = 2 * _r2[j];
      _Db[j] = abs(_C1[j] * _beta->get_params()[j] - current[i]);
      _X2[j] = _beta->get_params()[j] - _A1[j] * _Db[j];
      _A1[j] = 2 * _a * _r1 - _a;
      _C1[j] = 2 * _r2[j];
      _Dd[j] = abs(_C1[j] * _delta->get_params()[j] - current[i]);
      _X3[j] = _delta->get_params()[j] - _A1[j] * _Da[j];

      get_solutions()->get(j)->get_params()[j] = (_X1[j] + _X2[j] + _X3[j])/3;
    }
  }
}

solution* gwo::get_best()
{

}

void gwo::set_params(std::vector<float> &params)
{

}

void gwo::reset()
{

}

void gwo::update_elite()
{
  float size = get_solutions()->size();

  for(int i = 0; i < size; i++)
  {

    solution* current = get_solutions()->get(i);

    if (current->is_better_than(_alpha, is_maximization()))
    {
      _alpha = current;
    }
    else if (current->is_better_than(_beta, is_maximization()))
    {
      _beta = current;
    }
    else if (current->is_better_than(_delta, is_maximization()))
    {
      _delta = current;
    }
  }
}

void gwo::init()
{

  _alpha = get_solutions()->get(0);
  _beta = get_solutions()->get(1);
  _delta = get_solutions()->get(2);
  _dim=get_solutions()->get_dim();
  _a = 2.0f;
  _r1 = new float[2 * get_solutions()->get_dim()];
  _r2 = new float[2 * get_solutions()->get_dim()];
  _A1 = new float[2 * get_solutions()->get_dim()];
  _C1 = new float[2 * get_solutions()->get_dim()];
  _Da = new float[2 * get_solutions()->get_dim()];
  _Db = new float[2 * get_solutions()->get_dim()];
  _Dd = new float[2 * get_solutions()->get_dim()];
  _X1 = new float[2 * get_solutions()->get_dim()];
  _X2 = new float[2 * get_solutions()->get_dim()];
  _X3 = new float[2 * get_solutions()->get_dim()];
  _X =  new float[2 * get_solutions()->get_dim()];

}

gwo::~gwo()
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
