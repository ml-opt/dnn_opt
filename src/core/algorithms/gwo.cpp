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

void gwo::update_Elite()
{
  for(int i =0;i<=solution.size();i++){
    solution* current = get_solutions()->get(i);
    if (current->is_better_than(_alpha,is_maximization())) {
       _alpha=current;
    }
    else if (current->is_better_than(_beta,is_maximization())) {
      _beta=current;
    }
    else if (current->is_better_than(_delta,is_maximization())) {
     _delta=current;
    }
  }

}

void gwo::init()
{


}

gwo::~gwo()
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
