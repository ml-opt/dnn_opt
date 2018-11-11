#include <vector>
#include <stdexcept>
#include <copt/algorithms/cuckoo.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

void cuckoo::generate_new_cuckoo(int cuckoo_idx)
{
  auto cuckoo = get_solutions()->get(cuckoo_idx);
  auto best = get_solutions()->get_best(is_maximization());

  float v = _nd_1->generate();
  float u = _nd_o->generate();
  float levy = u / powf(fabs(v), 1 / _levy);

  _nd_1->generate(get_solutions()->get_dim(), _r);

  for(int i = 0; i < get_solutions()->get_dim(); i++)
  {
    float diff = best->get(i) - cuckoo->get(i);

    _updated->set(i, cuckoo->get(i) + _scale * levy * diff * _r[i]);
  }
}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt
