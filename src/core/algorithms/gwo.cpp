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
  _generator->generate(_dim, _r1);
  _generator->generate(_dim, _r2);
  for (int j = 0; j < _dim; j++)
  {
   _A[j] = 2 * _a * _r1 - _a;
   _C[j] = 2 * _r2[j];
  }
  for (int i = 0; i < get_solutions()->size(); i++)
  {
    auto* current = get_solutions()->get(i)->get_params();

    for (int j = 0; j < _dim; j++)
    {

      _Da[j] = abs(_C[j] * _alpha->get_params()[j] - current[i]);
      _X1[j] = _alpha->get_params()[j] - _A[j] * _Da[j];
      _Db[j] = abs(_C[j] * _beta->get_params()[j] - current[i]);
      _X2[j] = _beta->get_params()[j] - _A[j] * _Db[j];
      _Dd[j] = abs(_C[j] * _delta->get_params()[j] - current[i]);
      _X3[j] = _delta->get_params()[j] - _A[j] * _Da[j];

      current[j] = (_X1[j] + _X2[j] + _X3[j])/3;
    }
  }

  for (int j = 0; j < _dim; j++)
  {
   _a[j] -= 0.01;
  }
  update_elite();
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
  _dim = get_solutions()->get_dim();
  _a, _r1, _r2, _A, _C, _Da, _Db, _Dd, _X1, _X2, _X3 = new float[_dim];

  for (int j = 0; j < _dim; j++)
  {
   _a[j] = 2;
  }

}

gwo::~gwo()
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
