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
  for(int i = 0; i < size; i++){
    solution* current = get_solutions()->get(i);
    if (current->is_better_than(_alpha,is_maximization())) {
      _alpha = current;
    }
    else if (current->is_better_than(_beta,is_maximization())) {
      _beta = current;
    }
    else if (current->is_better_than(_delta,is_maximization())) {
      _delta = current;
    }
  }

}

void gwo::init()
{
  _alpha = get_solutions()->get(0);
  _beta = get_solutions()->get(1);
  _delta = get_solutions()->get(2);
  _a = 2.0;
  _r1= new float[2 * get_solutions()->get_dim()];
  _r2= new float[2 * get_solutions()->get_dim()];
  _A1= new float[2 * get_solutions()->get_dim()];
  _C1= new float[2 * get_solutions()->get_dim()];
}

gwo::~gwo()
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
